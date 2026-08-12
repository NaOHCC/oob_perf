import torch
import contextlib
import os
from torch.utils._python_dispatch import TorchDispatchMode
import time


def set_env():
    torch._C._set_sdp_use_mem_efficient(False)
    torch._C._set_sdp_use_flash(False)
    torch._C._set_sdp_use_overrideable(False)
    torch._C._set_sdp_use_math(True)


class DispatchLog(TorchDispatchMode):
    def __init__(self, model_name="", test_mode='eval', save_args=False, op_replay=False, enable_phase_tracking=False):
        super(DispatchLog, self).__init__()
        self.enable_phase_tracking = enable_phase_tracking
        self.flops = 0
        self.memory = 0
        self.flops_gemm_conv = 0
        self.memory_gemm_conv = 0
        self.memory_B580 = 0
        self.memory_gemm_conv_B580 = 0
        self.memory_G31 = 0
        self.memory_gemm_conv_G31 = 0
        self.memory_4080 = 0
        self.memory_gemm_conv_4080 = 0
        self.op_memory = 0
        self.op_memory_B580 = 0
        self.op_memory_G31 = 0
        self.op_memory_4080 = 0
        self.overlapped_flops_B580 = 0
        self.overlapped_memory_B580 = 0
        self.overlapped_flops_G31 = 0
        self.overlapped_memory_G31 = 0
        self.overlapped_flops_4080 = 0
        self.overlapped_memory_4080 = 0

        self.overlapped_gemm_conv_flops_B580 = 0
        self.overlapped_gemm_conv_memory_B580 = 0
        self.overlapped_gemm_conv_flops_G31 = 0
        self.overlapped_gemm_conv_memory_G31 = 0
        self.overlapped_gemm_conv_flops_4080 = 0
        self.overlapped_gemm_conv_memory_4080 = 0

        # Phase tracking for TTFT/TPOT
        if self.enable_phase_tracking:
            self.phase = 'prefill'
            self._sdpa_detected = False
            self.prefill_attention_count = 0
            self.decode_attention_count = 0
            self.prefill_metrics = DispatchLog._new_phase_metrics()
            self.decode_metrics = DispatchLog._new_phase_metrics()
            # Model-level hook (most reliable, set via register_model)
            self._model_hook_active = False
            self._forward_count = 0
            self._hook_handle = None

        self.save_args = save_args
        self.op_replay = op_replay
        self.saved_op_index = 0
        self.dict_file_name = {}
        self.dict_op_count = {}
        self.name_file_dict = []
        self.model_name = model_name
        self.test_mode = test_mode

        self.roofline_B580 = 93 * 1000 * 1000 * 1000 * 1000 / (410 * 1000 * 1000 * 1000) # OPs per byte
        self.roofline_4080 = 100.96 * 1000 * 1000 * 1000 * 1000 / (716.8 * 1000 * 1000 * 1000) # OPs per byte
        self.roofline_G31 = 154 * 1000 * 1000 * 1000 * 1000 / (532 * 1000 * 1000 * 1000) # OPs per byte

        # L2 cache bandwidth (GB/s) — used to model cached ops as non-free
        # When op memory fits in L2 cache, we scale it by (DRAM_BW / L2_BW) so that
        # overlapped_memory / DRAM_BW still gives the correct projected time.
        self.l2_bw_B580 = 2000  # GB/s, estimated for Xe2/BMG
        self.l2_bw_G31 = 2000   # GB/s, estimated for Xe2/BMG
        self.l2_bw_4080 = 3000  # GB/s, Ada Lovelace (conservative)
        self.dram_bw_B580 = 410  # GB/s
        self.dram_bw_G31 = 532   # GB/s
        self.dram_bw_4080 = 716.8  # GB/s

        if save_args:
            import pathlib
            # self.op_meta_dir = str(pathlib.Path.cwd()) + f'/op_meta/{test_mode}_{model_name}/'
            self.op_meta_dir = os.environ.get("OOB_LOG_DIR", "") + f'/op_meta/{test_mode}_{model_name}/'
            os.makedirs(self.op_meta_dir, exist_ok=True)
        self.op_memory = 0
        self.op_flops = 0
        self.op_args_stride = []
        self.op_kwargs_stride = []

    def _calc_tensor_memory(self, t):
        if not isinstance(t, torch.Tensor):
            return 0
        memory = t.numel() * t.element_size()
        for i in range(len(t.stride())):
            if t.stride(i) == 0:
                memory = memory / t.size(i)
        return memory

    def _calc_arg_memory(self, arg):
        """Calculate memory for a single argument, recursing into lists/tuples."""
        if isinstance(arg, torch.Tensor):
            return self._calc_tensor_memory(arg)
        elif isinstance(arg, (list, tuple)):
            return sum(self._calc_arg_memory(a) for a in arg)
        return 0

    def _calc_op_memory(self, args, kwargs, output):
        """Calculate total memory bytes for one op (inputs + kwargs + outputs).
        
        Pure function — does not mutate self. Returns raw memory bytes.
        """
        mem = 0
        # args (recurse into nested lists, e.g. aten::cat args=([t1,t2,...], dim))
        for arg in (args if isinstance(args, (list, tuple)) else (args,)):
            mem += self._calc_arg_memory(arg)
        # kwargs
        for v in (kwargs or {}).values():
            mem += self._calc_arg_memory(v)
        # output
        if isinstance(output, torch.Tensor):
            mem += self._calc_tensor_memory(output)
        elif isinstance(output, (list, tuple)):
            for o in output:
                mem += self._calc_tensor_memory(o)
        return mem

    def get_shape(self, x):
        """Return shape info for logging. Pure function — no side effects."""
        if isinstance(x, torch.Tensor):
            return x.shape
        if isinstance(x, (list, tuple)):
            shape_list = []
            for t in x:
                if isinstance(t, torch.Tensor):
                    shape_list.append(t.shape)
                else:
                    shape_list.append(type(t))
            return shape_list
        return type(x)

    def get_stride(self,x):
        if isinstance(x, torch.Tensor):
            return x.stride()
        if isinstance(x, list) or isinstance(x, tuple):
            shape_list = []
            for t in x:
                if isinstance(t,  torch.Tensor):
                    shape_list.append(t.stride())
            return shape_list
 
        return []

    def convert_to_meta_tensor(self, t):
        t_meta = torch.empty_like(t, device='meta')
        # for non-contiguous tensor
        out = torch.as_strided(t_meta, t.shape, t.stride())
        return out


    def convert_arg(self, arg):
        if isinstance(arg, torch.Tensor):
                return self.convert_to_meta_tensor(arg)
        elif isinstance(arg, list) or isinstance(arg, tuple):
            meta_ = []
            for a in arg:
                if isinstance(a, torch.Tensor):
                    meta_.append(self.convert_to_meta_tensor(a))
                else:
                    meta_.append(self.convert_arg(a))
            return meta_
        else:
            return arg
    
    def _get_flops_factor(self, tensor, op_name):
        supported_dtypes = {
            torch.float32: 2,
            torch.float16: 1,
            torch.bfloat16: 1,
        }
        dtype = tensor.dtype
        assert dtype in supported_dtypes, \
            f"{op_name} only supports FP16/BF16/FP32, got {dtype}"
        return supported_dtypes[dtype]

    @staticmethod
    def _new_phase_metrics():
        return {
            'flops': 0, 'memory': 0,
            'flops_gemm_conv': 0, 'memory_gemm_conv': 0,
            'memory_B580': 0, 'memory_G31': 0, 'memory_4080': 0,
            'memory_gemm_conv_B580': 0, 'memory_gemm_conv_G31': 0, 'memory_gemm_conv_4080': 0,
            'overlapped_flops_B580': 0, 'overlapped_memory_B580': 0,
            'overlapped_flops_G31': 0, 'overlapped_memory_G31': 0,
            'overlapped_flops_4080': 0, 'overlapped_memory_4080': 0,
            'overlapped_gemm_conv_flops_B580': 0, 'overlapped_gemm_conv_memory_B580': 0,
            'overlapped_gemm_conv_flops_G31': 0, 'overlapped_gemm_conv_memory_G31': 0,
            'overlapped_gemm_conv_flops_4080': 0, 'overlapped_gemm_conv_memory_4080': 0,
        }

    def register_model(self, model):
        """Register model forward hook for reliable phase detection.
        
        Uses model.register_forward_pre_hook to count forward calls:
        - 1st forward call = prefill
        - 2nd+ forward calls = decode
        
        This works for ALL models regardless of their internal op patterns.
        """
        if not self.enable_phase_tracking:
            return
        self._model_hook_active = True
        self._forward_count = 0

        def _forward_pre_hook(module, args):
            self._forward_count += 1
            if self._forward_count >= 2:
                self.phase = 'decode'

        self._hook_handle = model.register_forward_pre_hook(_forward_pre_hook)

    def unregister_model(self):
        """Remove the forward hook. Safe to call even if phase tracking is disabled."""
        hook = getattr(self, '_hook_handle', None)
        if hook is not None:
            hook.remove()
            self._hook_handle = None

    def _detect_phase(self, q_len):
        """Detect prefill->decode phase transition.
        
        q_len comes from:
        - embedding indices last dim (seq_len of input tokens)
        - attention query's seq dim (q_len)
        Both are > 1 during prefill, == 1 during decode.
        """
        if q_len > 1:
            if self.phase == 'prefill':
                self.prefill_attention_count += 1
        elif q_len == 1:
            if self.phase == 'prefill' and self.prefill_attention_count > 0:
                self.phase = 'decode'
            if self.phase == 'decode':
                self.decode_attention_count += 1

    def _accumulate_to_phase(self, is_mm_conv_op):
        """Accumulate current op metrics into the active phase bucket."""
        pm = self.prefill_metrics if self.phase == 'prefill' else self.decode_metrics
        pm['flops'] += self.op_flops
        pm['memory'] += self.op_memory
        pm['memory_B580'] += self.op_memory_B580
        pm['memory_G31'] += self.op_memory_G31
        pm['memory_4080'] += self.op_memory_4080

        for device, roofline in [('B580', self.roofline_B580), ('G31', self.roofline_G31), ('4080', self.roofline_4080)]:
            op_mem = getattr(self, f'op_memory_{device}')
            if op_mem == 0 or self.op_flops / op_mem > roofline:
                pm[f'overlapped_flops_{device}'] += self.op_flops
            else:
                pm[f'overlapped_memory_{device}'] += op_mem
            if is_mm_conv_op:
                if op_mem == 0 or self.op_flops / op_mem > roofline:
                    pm[f'overlapped_gemm_conv_flops_{device}'] += self.op_flops
                else:
                    pm[f'overlapped_gemm_conv_memory_{device}'] += op_mem

        if is_mm_conv_op:
            pm['flops_gemm_conv'] += self.op_flops
            pm['memory_gemm_conv'] += self.op_memory
            pm['memory_gemm_conv_B580'] += self.op_memory_B580
            pm['memory_gemm_conv_G31'] += self.op_memory_G31
            pm['memory_gemm_conv_4080'] += self.op_memory_4080

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        op_flops = self.flops
        op = func.name()

        # Phase detection for TTFT/TPOT (before any accumulation)
        # When model hook is active, phase is set by forward_pre_hook (most reliable).
        # Op-level detection is only used as fallback when no model is registered.
        if self.enable_phase_tracking and not self._model_hook_active:
            # Detect on embedding first (earliest signal in each forward pass)
            if "embedding" in op and isinstance(args, (list, tuple)) and len(args) >= 2:
                if isinstance(args[1], torch.Tensor):
                    if args[1].dim() >= 2:
                        self._detect_phase(args[1].shape[-1])
                    elif args[1].dim() == 1:
                        self._detect_phase(args[1].shape[0])
            # Then detect on attention ops
            elif "scaled_dot_product" in op and "backward" not in op:
                self._sdpa_detected = True
                if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].dim() >= 2:
                    self._detect_phase(args[0].shape[-2])
            elif op == "aten::bmm" and not self._sdpa_detected:
                if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].dim() >= 2:
                    self._detect_phase(args[0].shape[-2])

        DECOMPOSE_OPS = {
            "aten::matmul",
            "aten::linear",
        }
        if op in DECOMPOSE_OPS:
            with self:
                decomposed_result = func.decompose(*args, **(kwargs or {}))
                if decomposed_result is not NotImplemented:
                    return decomposed_result
        output = func(*args, **(kwargs or {}))
        # if isinstance(output, torch.Tensor):
        #     is_all_zero = torch.all(output == 0).item()
        # else:
        #     is_all_zero = "Not tensor, False"
        is_all_zero = False

        if self.save_args or self.op_replay:
            # filt the op of type memory, index related or record functions
            filter_op = ["view", "transpose", "slice", "split", "unsqueeze", "permute", "expand",
            "aten::select", "aten::t", "detach", "aten::_local_scalar_dense", "aten::lift_fresh",
            "record_function", "ones_like", "copy", "ones", "zeros"
            ]
            index_op = ["scatter", "gather", "embedding"]
            for i in filter_op:
                if i in op:
                    return output
            for i in index_op:
                if i in op:
                    return output
            if "clone" not in op: # cannot use torch save on args of clone
                if self.save_args:
                    args_meta = self.convert_arg(args)
                    kwargs_meta = self.convert_arg(kwargs)
                    tmp_result = [func._opname, func._overloadname, args_meta, kwargs_meta]
                    tmp_result_str = str(tmp_result)
                    if tmp_result_str not in self.dict_file_name:
                        tmp_file_name = f"{self.test_mode}_{op}_benchmark_{self.saved_op_index}.pt"
                        tmp_file_path = self.op_meta_dir + tmp_file_name
                        self.dict_file_name[tmp_result_str] = tmp_file_path
                        self.dict_op_count[tmp_file_path] = 1
                        try:
                            torch.save(tmp_result, tmp_file_path)
                            self.saved_op_index += 1

                        except Exception as e:
                            print(f"error saving op {str(tmp_result)}: {e}")
                    else:
                        tmp_file_path = self.dict_file_name[tmp_result_str]
                        self.dict_op_count[tmp_file_path] += 1

                if self.op_replay:
                    for _ in range(10):
                        func(*args, **(kwargs or {}))
                    torch.accelerator.synchronize()
                    start = time.time()
                    for _ in range(50):
                        func(*args, **(kwargs or {}))
                    torch.accelerator.synchronize()
                    end = time.time()
                    #print("benchmark", op, (end - start) / 50 * 1000, "ms")
        if "view" in op or "transpose" in op or "slice" in op or "split" in op or "unsqueeze" in op or "permute" in op or "expand" in op or "aten::select" in op or op == "aten::t" or "detach" in op or op == "aten::_local_scalar_dense" or op=="aten::lift_fresh" or op=="aten::as_strided" or op=="aten::as_strided_":
            return output
        if "reshape" in op:
            if len(args) > 0 and isinstance(args[0], torch.Tensor) and isinstance(output, torch.Tensor):
                try:
                    if output.untyped_storage().data_ptr() != args[0].untyped_storage().data_ptr():
                        pass 
                    else:
                        return output
                except RuntimeError:
                    return output
            else:
                return output

        embedding_skip_mem = 0
        if "embedding" in op and isinstance(args, (list, tuple)) and len(args) >= 2:
            if isinstance(args[0], torch.Tensor):
                embedding_skip_mem = self._calc_tensor_memory(args[0])

        args_shapes = self.get_shape(args) if isinstance(args, torch.Tensor) else  (tuple(self.get_shape(arg) for arg in args))
        kwargs_shapes = {k: self.get_shape(v) for k, v in (kwargs or {}).items()}
        args_strides = self.get_stride(args) if isinstance(args, torch.Tensor) else  (tuple(self.get_stride(arg) for arg in args))
        kwargs_strides = {k: self.get_stride(v) for k, v in (kwargs or {}).items()}
        if isinstance(output, torch.Tensor):
            output_shapes = self.get_shape(output)
        elif isinstance(output, list) or isinstance(output, tuple):
            output_shapes = tuple(self.get_shape(o) for o in output)

        # Memory accounting (decoupled from shape extraction)
        raw_op_memory = self._calc_op_memory(args, kwargs, output)
        if embedding_skip_mem > 0:
            raw_op_memory -= embedding_skip_mem

        # nll_loss_forward: input[B,C] is only gathered at B positions, not fully read
        # output is (scalar_loss, scalar_total_weight)
        if op == "aten::nll_loss_forward":
            input_tensor = args[0]  # logits [B, C]
            target_tensor = args[1]  # target [B]
            elem_size = input_tensor.element_size()
            batch_size = target_tensor.numel()
            gathered_input_mem = batch_size * elem_size
            target_mem = self._calc_tensor_memory(target_tensor)
            output_mem = 2 * elem_size  # loss scalar + total_weight scalar
            raw_op_memory = gathered_input_mem + target_mem + output_mem

        # nll_loss_backward: input[B,C] (logits) is NOT read; only grad_out, target are read
        # output grad_input[B,C] is fully written (zeroed + scattered)
        if op == "aten::nll_loss_backward":
            grad_output = args[0]  # scalar
            target_tensor = args[2]  # target [B]
            grad_input = output  # [B, C]
            elem_size = grad_input.element_size()
            grad_out_mem = grad_output.numel() * elem_size
            target_mem = self._calc_tensor_memory(target_tensor)
            grad_input_mem = self._calc_tensor_memory(grad_input)
            raw_op_memory = grad_out_mem + target_mem + grad_input_mem
        if op == "aten::bmm":
            factor = self._get_flops_factor(args[0], op)
            self.flops += factor * 2 * args[0].shape[0]* args[0].shape[1] * args[0].shape[2] * args[1].shape[2]
        if op == "aten::addmm":
            factor = self._get_flops_factor(args[0], op)
            self.flops += factor * (2 * args[1].shape[0]*args[1].shape[1] * args[2].shape[-1]+ args[1].shape[0]*args[2].shape[1])
        if op == "aten::mm":
            factor = self._get_flops_factor(args[0], op)
            self.flops += factor * (2 * args[0].shape[0]*args[0].shape[1] * args[1].shape[-1])
        if op == "aten::_grouped_mm":
            # Grouped matmul: out_i = mat_a_i @ mat_b_i (no transpose)
            #   2D: mat_a [S, K], mat_b [G, K, N], offs [G] — tokens grouped by offs
            #   3D: mat_a [G, T, K], mat_b [G, K, N], offs=None — pre-grouped
            # FLOPs = 2 * total_tokens * K * N
            input_t, weight_t = args[0], args[1]
            factor = self._get_flops_factor(input_t, op)
            if input_t.dim() == 3:
                total_tokens = input_t.shape[0] * input_t.shape[1]
            else:
                total_tokens = input_t.shape[0]
            self.flops += factor * 2 * total_tokens * weight_t.shape[1] * weight_t.shape[2]
        if op == "aten::_scaled_mm":
            # input1, input2, scale, bias, out_dtype
            input1, input2 = args[0], args[1]
            bias = args[3] if len(args) > 3 else None
            # A @ B
            flops_matmul = 2 * input1.numel() * input2.shape[-1]
            self.flops += flops_matmul
        if "scaled_dot_product" in op and "backward" not in op:
            # query, key, value
            q, k, v = args[0], args[1], args[2]
            attn_mask = args[3] if len(args) > 3 and args[3] is not None else None
            q_len, kv_len = q.shape[-2], k.shape[-2]
            BxH = q.numel() // (q.shape[-1] * q_len)
            num_attn_scores = BxH * q_len * kv_len
            is_causal = False
            # is_causal position depends on backend:
            #   _fused_attention_overrideable(q, k, v, attn_mask, dropout_p, is_causal) → args[5]
            #   _flash_attention(q, k, v, dropout_p, is_causal, ...)                   → args[4]
            #   _cudnn_attention(q, k, v, ...)                                          → varies
            # Strategy: check by op name first, then fallback scan args for bool
            if "overrideable" in op and len(args) > 5 and isinstance(args[5], bool):
                is_causal = args[5]
            elif "flash" in op and len(args) > 4 and isinstance(args[4], bool):
                is_causal = args[4]
            else:
                # Fallback: scan args from the end for a bool (skip tensors/None/float)
                for i in range(len(args) - 1, 2, -1):
                    if isinstance(args[i], bool):
                        is_causal = args[i]
                        break
            if not is_causal and "is_causal" in (kwargs or {}) and kwargs["is_causal"] is not None:
                is_causal = kwargs["is_causal"]

            if is_causal and q_len > 1:
                # L*(L+1)/2 replace L*L
                num_attn_scores = q_len * (q_len + 1) // 2 * BxH

            # Q @ K^T
            flops_qk_t = 2 * num_attn_scores * q.shape[-1]
            self.flops += flops_qk_t

            # scores @ V
            flops_attn_v = 2 * num_attn_scores * v.shape[-1]
            self.flops += flops_attn_v
            # flash_attn library ops (flash_attn::_flash_attn_forward, flash_attn::_flash_attn_varlen_forward)
        if "flash_attn::_flash_attn_forward" in op and "varlen" not in op:
            # q: [batch, seq_q, heads_q, head_dim], k: [batch, seq_k, heads_k, head_dim]
            q, k, v = args[0], args[1], args[2]
            batch, seq_q, heads_q, head_dim = q.shape
            seq_k = k.shape[1]
            # Scan args forward for is_causal (first bool after tensors)
            # Signature: q, k, v, dropout_p, softmax_scale, causal, window_size_left, window_size_right, softcap, return_softmax
            is_causal = False
            for i in range(3, len(args)):
                if isinstance(args[i], bool):
                    is_causal = args[i]
                    break
            num_attn_pairs = batch * heads_q * seq_q * seq_k
            if is_causal and seq_q == seq_k and seq_q > 1:
                num_attn_pairs = batch * heads_q * seq_q * (seq_q + 1) // 2
            # Q@K^T + scores@V
            self.flops += 2 * num_attn_pairs * head_dim  # Q@K^T
            self.flops += 2 * num_attn_pairs * head_dim  # scores@V
        if "flash_attn::_flash_attn_varlen_forward" in op:
            # q: [total_q, heads_q, head_dim], k: [total_k, heads_k, head_dim]
            # cu_seqlens_q: args[3], cu_seqlens_k: args[4]
            # max_seqlen_q: args[5], max_seqlen_k: args[6]
            # dropout_p: args[7], softmax_scale: args[8], is_causal: args[9]
            q, k, v = args[0], args[1], args[2]
            cu_seqlens_q, cu_seqlens_k = args[3], args[4]
            is_causal = args[9] if len(args) > 9 and isinstance(args[9], bool) else False
            heads_q = q.shape[1]
            head_dim = q.shape[2]
            # Compute per-sequence FLOPs from cu_seqlens
            cu_q = cu_seqlens_q.cpu()
            cu_k = cu_seqlens_k.cpu()
            num_seqs = cu_q.shape[0] - 1
            total_attn_pairs = 0
            for i in range(num_seqs):
                sq = int(cu_q[i+1] - cu_q[i])
                sk = int(cu_k[i+1] - cu_k[i])
                if is_causal and sq == sk and sq > 1:
                    total_attn_pairs += sq * (sq + 1) // 2
                else:
                    total_attn_pairs += sq * sk
            # Q@K^T + scores@V
            self.flops += 2 * heads_q * total_attn_pairs * head_dim  # Q@K^T
            self.flops += 2 * heads_q * total_attn_pairs * head_dim  # scores@V
        if "scaled_dot_product" in op and "backward" in op:
            # Backward ops: _scaled_dot_product_{flash,efficient,cudnn}_attention_backward
            #               _scaled_dot_product_fused_attention_overrideable_backward
            # All have signature: (grad_out, query, key, value, ...)
            grad_out, query, key, value = args[0], args[1], args[2], args[3]
            batch, num_heads_q, seq_len_q, head_dim_q = query.shape
            seq_len_k = key.shape[-2]
            head_dim_v = value.shape[-1]
            batch_x_heads = batch * num_heads_q

            # Step 1: Recompute scores — query: [B,H,Sq,Dq] @ key^T: [B,H,Dq,Sk]
            self.flops += 2 * batch_x_heads * seq_len_q * seq_len_k * head_dim_q
            # Step 2a: grad_out @ value^T -> grad_scores — [B,H,Sq,Dv] @ [B,H,Dv,Sk]    
            self.flops += 2 * batch_x_heads * seq_len_q * seq_len_k * head_dim_v
            # Step 2b: scores^T @ grad_out -> grad_value — [B,H,Sk,Sq] @ [B,H,Sq,Dv]
            self.flops += 2 * batch_x_heads * seq_len_k * seq_len_q * head_dim_v
            # Step 3a: grad_scores @ key -> grad_query — [B,H,Sq,Sk] @ [B,H,Sk,Dq]
            self.flops += 2 * batch_x_heads * seq_len_q * seq_len_k * head_dim_q
            # Step 3b: query^T @ grad_scores -> grad_key — [B,H,Dq,Sq] @ [B,H,Sq,Sk]
            self.flops += 2 * batch_x_heads * head_dim_q * seq_len_q * seq_len_k
        if op == "aten::convolution":
            ndim = args[0].dim()
            assert ndim in (3, 4, 5) and args[1].dim() == ndim, \
                f"Unsupported convolution dims: input={args[0].dim()}, weight={args[1].dim()}"
            transposed = args[6]
            if type(transposed) is not bool:
                for arg in args:
                    if type(arg) is bool:
                        transposed = arg
                        break
            assert type(transposed) is bool
            group = args[-1]
            assert type(group) is int

            N = args[0].shape[0]
            C_i = args[0].shape[1]
            C_o = output.shape[1]
            K_1 = args[1].shape[1]
            # Compute spatial product for input, kernel, and output
            # conv1d: 1 spatial dim, conv2d: 2, conv3d: 3
            import math
            spatial_in = math.prod(args[0].shape[2:])
            spatial_k = math.prod(args[1].shape[2:])
            spatial_out = math.prod(output.shape[2:])
            if transposed:
                self.flops += 2 * spatial_in * N * spatial_k * K_1 * C_o / group
            else:
                self.flops += 2 * spatial_out * N * spatial_k * K_1 * C_o / group
            if isinstance(args[2], torch.Tensor):
                # bias is another kernel
                raw_op_memory += N * C_o * spatial_out * 2
                self.flops += N * C_o * spatial_out
        if op == "aten::convolution_backward":
            transposed = args[6]
            if type(transposed) is not bool:
                for arg in args:
                    if type(arg) is bool:
                        transposed = arg
                        break
            assert type(transposed) is bool
            # output = (grad_input, grad_weight, grad_bias)
            # input = (grad_output, input, weight)
            ndim = args[1].dim()
            assert ndim in (3, 4, 5) and args[2].dim() == ndim, \
                f"Unsupported conv_backward dims: input={args[1].dim()}, weight={args[2].dim()}"
            group = args[-2]
            assert type(group) is int

            N = args[1].shape[0]
            C_i = args[1].shape[1]
            C_o = args[0].shape[1]
            import math
            output_mask = args[-1]
            spatial_in = math.prod(args[1].shape[2:])
            spatial_k = math.prod(args[2].shape[2:])
            spatial_out = math.prod(args[0].shape[2:])
            if output_mask[0]:
                #grad_input
                if transposed:
                    self.flops += 2 * spatial_out * N * spatial_k * C_i * C_o / group
                else:
                    self.flops += 2 * spatial_in * N * spatial_k * C_i * C_o / group
            if output_mask[1]:
                if transposed:
                    self.flops += 2 * spatial_in * N * spatial_k * C_i * C_o / group
                else:
                    self.flops += 2 * spatial_out * N * spatial_k * C_i * C_o / group

        # Vector engine ops: max_pool, batch_norm, layer_norm, softmax
        # These run on the vector engine, not the matrix engine. Since our peak TFLOPS
        # is for the matrix engine, counting their FLOPs would misclassify them as
        # compute-bound. They are memory-bound in practice, so set FLOPs to 0.
        # if op == "aten::max_pool2d_with_indices":
        #     N = output[0].shape[0]
        #     C = output[0].shape[1]
        #     H = output[0].shape[2]
        #     W = output[0].shape[3]
        #     K_h = args[1][0]
        #     K_w = args[1][-1]
        #     self.flops += N * C * H * W * (K_h * K_w - 1)
        # if "batch_norm" in op:
        #     self.flops += 4*args[0].numel()
        # if op == "aten::native_layer_norm":
        #     if args[0].dtype == torch.float16:
        #         self.flops += 5*args[0].numel()
        #     else:
        #         self.flops += 5*args[0].numel() * 15 # suppose fp32 TFLOPS is 15x slower than FP16 according to spec
        # if op == "aten::_softmax":
        #     self.flops += 4*args[0].numel()
        self.op_flops = self.flops - op_flops
        # self.op_memory = self.memory - op_memory
        self.memory += raw_op_memory
        self.op_memory = raw_op_memory

        # For inference, weight tensors are NOT in L2 cache (total model params >> L2).
        # Identify weight memory so it always uses DRAM bandwidth (no cache discount).
        weight_memory = 0
        if self.test_mode in ('eval', 'inference'):
            if op == "aten::mm":
                # mm(activation, weight): weight = args[1]
                weight_memory = self._calc_tensor_memory(args[1])
            elif op == "aten::addmm":
                # addmm(bias, activation, weight): weight = args[2], bias = args[0]
                weight_memory = self._calc_tensor_memory(args[2]) + self._calc_tensor_memory(args[0])
            elif op == "aten::_scaled_mm":
                # _scaled_mm(input, weight, ...): weight = args[1]
                weight_memory = self._calc_tensor_memory(args[1])
            elif op == "aten::convolution":
                # convolution(input, weight, bias, ...): weight = args[1]
                weight_memory = self._calc_tensor_memory(args[1])
                if len(args) > 2 and isinstance(args[2], torch.Tensor):
                    weight_memory += self._calc_tensor_memory(args[2])
            elif op == "aten::_grouped_mm":
                # mat_b [G, K, N]: only active experts' weights are read
                weight_t = args[1]
                offs_t = args[2] if len(args) > 2 else None
                per_expert_bytes = weight_t.shape[1] * weight_t.shape[2] * weight_t.element_size()
                if offs_t is None or not isinstance(offs_t, torch.Tensor):
                    # 3D input: offs=None, all G experts active
                    active_experts = weight_t.shape[0]
                else:
                    # 2D input: count experts with non-zero group size from offs
                    offs_cpu = offs_t.detach().cpu().tolist()
                    active_experts = sum(
                        1 for i in range(len(offs_cpu))
                        if offs_cpu[i] > (offs_cpu[i-1] if i > 0 else 0)
                    )
                weight_memory = active_experts * per_expert_bytes
        
        for threshold, attr_mem, attr_op, dram_bw, l2_bw in [
            (18 * 1000 * 1000, 'memory_B580', 'op_memory_B580', self.dram_bw_B580, self.l2_bw_B580),
            (24 * 1000 * 1000, 'memory_G31', 'op_memory_G31', self.dram_bw_G31, self.l2_bw_G31),
            (64 * 1000 * 1000, 'memory_4080', 'op_memory_4080', self.dram_bw_4080, self.l2_bw_4080)
        ]:
            bw_ratio = dram_bw / l2_bw  # < 1, so cached portion costs less than DRAM
            if weight_memory > 0:
                # Inference mm/conv: weight at DRAM rate, rest with cache logic
                non_weight = raw_op_memory - weight_memory
                if non_weight > 0 and non_weight <= threshold:
                    setattr(self, attr_op, weight_memory + non_weight * bw_ratio)
                else:
                    # non_weight also exceeds cache or is zero — all at DRAM rate
                    setattr(self, attr_op, raw_op_memory)
            elif raw_op_memory > threshold:
                # Part exceeding cache: full DRAM cost; cached part: scaled by bw_ratio
                setattr(self, attr_op, (raw_op_memory - threshold) + threshold * bw_ratio)
            else:
                # All fits in cache: not free, but costs less (scaled by DRAM/L2 ratio)
                setattr(self, attr_op, raw_op_memory * bw_ratio)
            setattr(self, attr_mem, getattr(self, attr_mem) + getattr(self, attr_op))
        
        self.op_args_stride = args_strides
        self.op_kwargs_stride = kwargs_strides
        is_mm_conv_op = (
            op in ("aten::bmm", "aten::mm", "aten::addmm", "aten::_grouped_mm",
                "aten::convolution", "aten::convolution_backward")
            or
            ("scaled_dot_product" in op)
            or
            ("flash_attn::_flash_attn" in op)
        )
        # delta_flops = self.flops - op_flops
        # delta_memory = self.memory - op_memory

        if is_mm_conv_op:
            self.flops_gemm_conv += self.op_flops
            self.memory_gemm_conv += self.op_memory
            self.memory_gemm_conv_B580 += self.op_memory_B580
            self.memory_gemm_conv_G31 += self.op_memory_G31
            self.memory_gemm_conv_4080 += self.op_memory_4080

            if self.op_memory_B580 == 0 or self.op_flops / self.op_memory_B580 > self.roofline_B580:
                self.overlapped_gemm_conv_flops_B580 += self.op_flops
            else:
                self.overlapped_gemm_conv_memory_B580 += self.op_memory_B580
            
            if self.op_memory_G31 == 0 or self.op_flops / self.op_memory_G31 > self.roofline_G31:
                self.overlapped_gemm_conv_flops_G31 += self.op_flops
            else:
                self.overlapped_gemm_conv_memory_G31 += self.op_memory_G31
            
            if self.op_memory_4080 == 0 or self.op_flops / self.op_memory_4080 > self.roofline_4080:
                self.overlapped_gemm_conv_flops_4080 += self.op_flops
            else:
                self.overlapped_gemm_conv_memory_4080 += self.op_memory_4080

        if self.op_memory_B580 == 0 or self.op_flops / self.op_memory_B580 > self.roofline_B580:
            self.overlapped_flops_B580 += self.op_flops
        else:
            self.overlapped_memory_B580 += self.op_memory_B580
        
        if self.op_memory_G31 == 0 or self.op_flops / self.op_memory_G31 > self.roofline_G31:
            self.overlapped_flops_G31 += self.op_flops
        else:
            self.overlapped_memory_G31 += self.op_memory_G31
        
        if self.op_memory_4080 == 0 or self.op_flops / self.op_memory_4080 > self.roofline_4080:
            self.overlapped_flops_4080 += self.op_flops
        else:
            self.overlapped_memory_4080 += self.op_memory_4080

        # Accumulate to phase-level metrics (prefill or decode)
        if self.enable_phase_tracking:
            self._accumulate_to_phase(is_mm_conv_op)

        # print(f"{op}|{self.flops}|{self.memory}|{self.flops_gemm_conv}|{self.memory_gemm_conv}|{args_shapes}|{is_all_zero}", flush=True)
        print(f"{op}|{self.flops}|{self.memory}|"
              f"{self.flops_gemm_conv}|{self.memory_gemm_conv}|"
              f"{self.memory_B580}|{self.memory_4080}|{self.memory_G31}|"
              f"{self.memory_gemm_conv_B580}|{self.memory_gemm_conv_4080}|{self.memory_gemm_conv_G31}|"
              f"{self.overlapped_flops_B580}|{self.overlapped_flops_4080}|{self.overlapped_flops_G31}|"
              f"{self.overlapped_memory_B580}|{self.overlapped_memory_4080}|{self.overlapped_memory_G31}|"
              f"{self.overlapped_gemm_conv_flops_B580}|{self.overlapped_gemm_conv_flops_4080}|{self.overlapped_gemm_conv_flops_G31}|"
              f"{self.overlapped_gemm_conv_memory_B580}|{self.overlapped_gemm_conv_memory_4080}|{self.overlapped_gemm_conv_memory_G31}|"
              f"args:{args_shapes}|zero:{is_all_zero}", flush=True)
        return output

def update_counts_to_files(dict_file_path):
    for file_name, count in dict_file_path.items():
        data = torch.load(file_name)
        data.append(count)
        torch.save(data, file_name)


@contextlib.contextmanager
def context_func(profiling_enabled, model_name, test_mode, device, fuser_mode='none', schedule_disable='no', total_iter=None):
    calculate_flops = os.environ.get("Calculate_Flops", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
    save_args = os.environ.get("SAVE_ARGS", "OFF").upper() in ["1", "Y", "ON", "YES", "True"]
    op_replay = os.environ.get("OP_REPLAY", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
    enable_phase_tracking = os.environ.get("PHASE_TRACKING", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
    
    profile_activity = [torch.profiler.ProfilerActivity.CPU]
    if device == "xpu":
        profile_activity.append(torch.profiler.ProfilerActivity.XPU)
    elif device == "cuda":
        profile_activity.append(torch.profiler.ProfilerActivity.CUDA)

    if schedule_disable == "yes":
        schedule = None
    elif total_iter != None:
        middle_iter = total_iter // 2 + 4 if total_iter >= 10 else total_iter
        schedule = torch.profiler.schedule(wait=middle_iter-3, warmup=3, active=1)
    else:
        schedule = torch.profiler.schedule(wait=6, warmup=3, active=1)

    if profiling_enabled:
        with torch.profiler.profile(activities=profile_activity, schedule=schedule, with_stack=True, record_shapes=True, with_modules=True, experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
            yield prof
    elif calculate_flops:
        if save_args:
            with DispatchLog(model_name, test_mode, save_args, op_replay, enable_phase_tracking) as calculator:
                yield calculator
            update_counts_to_files(calculator.dict_op_count)
        else:
            with DispatchLog(test_mode=test_mode, enable_phase_tracking=enable_phase_tracking) as calculator:
                yield calculator
            # Clean up model hook if registered
            calculator.unregister_model()
            print("memory:", calculator.memory, flush=True)
            print("flops:",calculator.flops, flush=True)
            print("memory_gemm_conv:", calculator.memory_gemm_conv, flush=True)
            print("flops_gemm_conv:", calculator.flops_gemm_conv, flush=True)
            
            print("memory_B580:", calculator.memory_B580, flush=True)
            print("memory_G31:", calculator.memory_G31, flush=True)
            print("memory_4080:", calculator.memory_4080, flush=True)
            print("memory_gemm_conv_B580:", calculator.memory_gemm_conv_B580, flush=True)
            print("memory_gemm_conv_G31:", calculator.memory_gemm_conv_G31, flush=True)
            print("memory_gemm_conv_4080:", calculator.memory_gemm_conv_4080, flush=True)

            print("overlapped_flops_B580:", calculator.overlapped_flops_B580, flush=True)
            print("overlapped_flops_G31:", calculator.overlapped_flops_G31, flush=True)
            print("overlapped_flops_4080:", calculator.overlapped_flops_4080, flush=True)
            print("overlapped_memory_B580:", calculator.overlapped_memory_B580, flush=True)
            print("overlapped_memory_G31:", calculator.overlapped_memory_G31, flush=True)
            print("overlapped_memory_4080:", calculator.overlapped_memory_4080, flush=True)

            print("overlapped_gemm_conv_flops_B580:", calculator.overlapped_gemm_conv_flops_B580, flush=True)
            print("overlapped_gemm_conv_flops_G31:", calculator.overlapped_gemm_conv_flops_G31, flush=True)
            print("overlapped_gemm_conv_flops_4080:", calculator.overlapped_gemm_conv_flops_4080, flush=True)
            print("overlapped_gemm_conv_memory_B580:", calculator.overlapped_gemm_conv_memory_B580, flush=True)
            print("overlapped_gemm_conv_memory_G31:", calculator.overlapped_gemm_conv_memory_G31, flush=True)
            print("overlapped_gemm_conv_memory_4080:", calculator.overlapped_gemm_conv_memory_4080, flush=True)

            # Phase-level metrics for TTFT/TPOT (printed together with E2E)
            if enable_phase_tracking:
                # Determine decode_steps based on detection method
                if calculator._model_hook_active:
                    # Model hook: forward_count - 1 = decode steps
                    decode_steps = calculator._forward_count - 1 if calculator._forward_count > 1 else 0
                    has_decode = decode_steps > 0
                else:
                    # Op-level fallback: use attention counts
                    num_layers = calculator.prefill_attention_count
                    decode_steps = calculator.decode_attention_count // num_layers if num_layers > 0 else 0
                    has_decode = calculator.decode_attention_count > 0

                if has_decode:
                    pm = calculator.prefill_metrics
                    dm = calculator.decode_metrics

                    print(f"phase_info: decode_steps={decode_steps}, detection={'model_hook' if calculator._model_hook_active else 'op_level'}", flush=True)

                    # Prefill (TTFT) overlapped metrics
                    print("prefill_flops:", pm['flops'], flush=True)
                    print("prefill_memory:", pm['memory'], flush=True)
                    print("prefill_flops_gemm_conv:", pm['flops_gemm_conv'], flush=True)
                    print("prefill_memory_gemm_conv:", pm['memory_gemm_conv'], flush=True)
                    print("prefill_memory_B580:", pm['memory_B580'], flush=True)
                    print("prefill_memory_G31:", pm['memory_G31'], flush=True)
                    print("prefill_memory_4080:", pm['memory_4080'], flush=True)
                    print("prefill_memory_gemm_conv_B580:", pm['memory_gemm_conv_B580'], flush=True)
                    print("prefill_memory_gemm_conv_G31:", pm['memory_gemm_conv_G31'], flush=True)
                    print("prefill_memory_gemm_conv_4080:", pm['memory_gemm_conv_4080'], flush=True)
                    print("prefill_overlapped_flops_B580:", pm['overlapped_flops_B580'], flush=True)
                    print("prefill_overlapped_flops_G31:", pm['overlapped_flops_G31'], flush=True)
                    print("prefill_overlapped_flops_4080:", pm['overlapped_flops_4080'], flush=True)
                    print("prefill_overlapped_memory_B580:", pm['overlapped_memory_B580'], flush=True)
                    print("prefill_overlapped_memory_G31:", pm['overlapped_memory_G31'], flush=True)
                    print("prefill_overlapped_memory_4080:", pm['overlapped_memory_4080'], flush=True)
                    print("prefill_overlapped_gemm_conv_flops_B580:", pm['overlapped_gemm_conv_flops_B580'], flush=True)
                    print("prefill_overlapped_gemm_conv_flops_G31:", pm['overlapped_gemm_conv_flops_G31'], flush=True)
                    print("prefill_overlapped_gemm_conv_flops_4080:", pm['overlapped_gemm_conv_flops_4080'], flush=True)
                    print("prefill_overlapped_gemm_conv_memory_B580:", pm['overlapped_gemm_conv_memory_B580'], flush=True)
                    print("prefill_overlapped_gemm_conv_memory_G31:", pm['overlapped_gemm_conv_memory_G31'], flush=True)
                    print("prefill_overlapped_gemm_conv_memory_4080:", pm['overlapped_gemm_conv_memory_4080'], flush=True)

                    # Decode total overlapped metrics
                    print("decode_flops:", dm['flops'], flush=True)
                    print("decode_memory:", dm['memory'], flush=True)
                    print("decode_flops_gemm_conv:", dm['flops_gemm_conv'], flush=True)
                    print("decode_memory_gemm_conv:", dm['memory_gemm_conv'], flush=True)
                    print("decode_memory_B580:", dm['memory_B580'], flush=True)
                    print("decode_memory_G31:", dm['memory_G31'], flush=True)
                    print("decode_memory_4080:", dm['memory_4080'], flush=True)
                    print("decode_memory_gemm_conv_B580:", dm['memory_gemm_conv_B580'], flush=True)
                    print("decode_memory_gemm_conv_G31:", dm['memory_gemm_conv_G31'], flush=True)
                    print("decode_memory_gemm_conv_4080:", dm['memory_gemm_conv_4080'], flush=True)
                    print("decode_overlapped_flops_B580:", dm['overlapped_flops_B580'], flush=True)
                    print("decode_overlapped_flops_G31:", dm['overlapped_flops_G31'], flush=True)
                    print("decode_overlapped_flops_4080:", dm['overlapped_flops_4080'], flush=True)
                    print("decode_overlapped_memory_B580:", dm['overlapped_memory_B580'], flush=True)
                    print("decode_overlapped_memory_G31:", dm['overlapped_memory_G31'], flush=True)
                    print("decode_overlapped_memory_4080:", dm['overlapped_memory_4080'], flush=True)
                    print("decode_overlapped_gemm_conv_flops_B580:", dm['overlapped_gemm_conv_flops_B580'], flush=True)
                    print("decode_overlapped_gemm_conv_flops_G31:", dm['overlapped_gemm_conv_flops_G31'], flush=True)
                    print("decode_overlapped_gemm_conv_flops_4080:", dm['overlapped_gemm_conv_flops_4080'], flush=True)
                    print("decode_overlapped_gemm_conv_memory_B580:", dm['overlapped_gemm_conv_memory_B580'], flush=True)
                    print("decode_overlapped_gemm_conv_memory_G31:", dm['overlapped_gemm_conv_memory_G31'], flush=True)
                    print("decode_overlapped_gemm_conv_memory_4080:", dm['overlapped_gemm_conv_memory_4080'], flush=True)

                    # Decode per-step (TPOT) overlapped metrics
                    if decode_steps > 0:
                        print("decode_flops_per_step:", dm['flops'] / decode_steps, flush=True)
                        print("decode_memory_per_step:", dm['memory'] / decode_steps, flush=True)
                        print("decode_flops_gemm_conv_per_step:", dm['flops_gemm_conv'] / decode_steps, flush=True)
                        print("decode_memory_gemm_conv_per_step:", dm['memory_gemm_conv'] / decode_steps, flush=True)
                        print("decode_memory_B580_per_step:", dm['memory_B580'] / decode_steps, flush=True)
                        print("decode_memory_G31_per_step:", dm['memory_G31'] / decode_steps, flush=True)
                        print("decode_memory_4080_per_step:", dm['memory_4080'] / decode_steps, flush=True)
                        print("decode_memory_gemm_conv_B580_per_step:", dm['memory_gemm_conv_B580'] / decode_steps, flush=True)
                        print("decode_memory_gemm_conv_G31_per_step:", dm['memory_gemm_conv_G31'] / decode_steps, flush=True)
                        print("decode_memory_gemm_conv_4080_per_step:", dm['memory_gemm_conv_4080'] / decode_steps, flush=True)
                        print("decode_overlapped_flops_B580_per_step:", dm['overlapped_flops_B580'] / decode_steps, flush=True)
                        print("decode_overlapped_flops_G31_per_step:", dm['overlapped_flops_G31'] / decode_steps, flush=True)
                        print("decode_overlapped_flops_4080_per_step:", dm['overlapped_flops_4080'] / decode_steps, flush=True)
                        print("decode_overlapped_memory_B580_per_step:", dm['overlapped_memory_B580'] / decode_steps, flush=True)
                        print("decode_overlapped_memory_G31_per_step:", dm['overlapped_memory_G31'] / decode_steps, flush=True)
                        print("decode_overlapped_memory_4080_per_step:", dm['overlapped_memory_4080'] / decode_steps, flush=True)
                        print("decode_overlapped_gemm_conv_flops_B580_per_step:", dm['overlapped_gemm_conv_flops_B580'] / decode_steps, flush=True)
                        print("decode_overlapped_gemm_conv_flops_G31_per_step:", dm['overlapped_gemm_conv_flops_G31'] / decode_steps, flush=True)
                        print("decode_overlapped_gemm_conv_flops_4080_per_step:", dm['overlapped_gemm_conv_flops_4080'] / decode_steps, flush=True)
                        print("decode_overlapped_gemm_conv_memory_B580_per_step:", dm['overlapped_gemm_conv_memory_B580'] / decode_steps, flush=True)
                        print("decode_overlapped_gemm_conv_memory_G31_per_step:", dm['overlapped_gemm_conv_memory_G31'] / decode_steps, flush=True)
                        print("decode_overlapped_gemm_conv_memory_4080_per_step:", dm['overlapped_gemm_conv_memory_4080'] / decode_steps, flush=True)
                else:
                    print("phase_info: no decode phase detected, all metrics are E2E only", flush=True)
    else:
        with contextlib.nullcontext(None):
            yield

    if profiling_enabled:
        save_profile(prof, device)
        print("---- save profile success")

def save_profile(prof, device):
    import pathlib
    import os
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    torch.save(prof.key_averages().table(sort_by="self_{}_time_total".format(device), row_limit=100000),
        timeline_dir+'profile.pt')
    torch.save(prof.key_averages(group_by_input_shape=True).table(),
        timeline_dir+'profile_detail.pt')
    #torch.save(prof.key_averages().table(sort_by="id", row_limit=100000),
    #    timeline_dir+'profile_detail_withId.pt')
    prof.export_chrome_trace(timeline_dir+"trace.json")
 