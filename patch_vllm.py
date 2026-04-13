"""
vLLM 0.19.0 patches for NanoNemotronVLProcessor compatibility.

Five crash sites in the multimodal dummy-input path; all idempotent
(skip if already applied or if the target string is not found).

Applied at Docker build time so the image always has the fixes regardless
of whether EXTRA_COMMANDS runs at instance launch time.
"""
import glob
import sys

ROOTS = ['/opt', '/usr', '/root', '/venv']


def _find(name):
    for r in ROOTS:
        for p in glob.glob(r + '/**/' + name, recursive=True):
            return p
    return None


def _patch(path, old, new, label):
    if not path:
        print(f'SKIP (not found): {label}', flush=True)
        return
    src = open(path).read()
    if old not in src:
        print(f'SKIP (already patched or mismatch): {label}', flush=True)
        return
    open(path, 'w').write(src.replace(old, new, 1))
    print(f'patched: {label}', flush=True)


# ── Patch 1: encoder_budget.py ──────────────────────────────────────────────
_patch(
    _find('encoder_budget.py'),
    (
        '    mm_inputs = mm_registry.get_dummy_mm_inputs(\n'
        '        model_config,\n'
        '        mm_counts=mm_counts,\n'
        '        processor=processor,\n'
        '    )\n'
        '\n'
        '    return {\n'
        '        modality: sum(item.get_num_embeds() for item in placeholders)\n'
        '        for modality, placeholders in mm_inputs["mm_placeholders"].items()\n'
        '    }'
    ),
    (
        '    try:\n'
        '        mm_inputs = mm_registry.get_dummy_mm_inputs(\n'
        '            model_config,\n'
        '            mm_counts=mm_counts,\n'
        '            processor=processor,\n'
        '        )\n'
        '    except Exception as _exc:\n'
        '        import logging as _l\n'
        '        _l.getLogger(__name__).warning(\n'
        '            "get_dummy_mm_inputs failed for %s (%s); "\n'
        '            "falling back to max_model_len=%d per modality",\n'
        '            type(processor).__name__, _exc, model_config.max_model_len,\n'
        '        )\n'
        '        return {m: model_config.max_model_len for m in mm_counts}\n'
        '\n'
        '    return {\n'
        '        modality: sum(item.get_num_embeds() for item in placeholders)\n'
        '        for modality, placeholders in mm_inputs["mm_placeholders"].items()\n'
        '    }'
    ),
    'encoder_budget.py',
)

# ── Patch 2: transformers_utils/processor.py ────────────────────────────────
_proc = None
for _r in ROOTS:
    for _p in glob.glob(_r + '/**/transformers_utils/processor.py', recursive=True):
        _proc = _p
        break
_patch(
    _proc,
    (
        '    output_kwargs = processor._merge_kwargs(\n'
        '        get_processor_kwargs_type(processor),\n'
        '        **kwargs,\n'
        '    )'
    ),
    (
        '    if hasattr(processor, "_merge_kwargs"):\n'
        '        output_kwargs = processor._merge_kwargs(\n'
        '            get_processor_kwargs_type(processor),\n'
        '            **kwargs,\n'
        '        )\n'
        '    else:\n'
        '        _mk = {"text_kwargs", "audio_kwargs", "images_kwargs",\n'
        '               "videos_kwargs", "cross_attention_kwargs"}\n'
        '        _c = {k: v for k, v in kwargs.items() if k not in _mk}\n'
        '        output_kwargs = {\n'
        '            "audio_kwargs":  {**_c, **kwargs.get("audio_kwargs",  {})},\n'
        '            "images_kwargs": {**_c, **kwargs.get("images_kwargs", {})},\n'
        '            "videos_kwargs": {**_c, **kwargs.get("videos_kwargs", {})},\n'
        '        }'
    ),
    'transformers_utils/processor.py',
)

# ── Patch 3: v1/worker/gpu_model_runner.py ───────────────────────────────────
_patch(
    _find('gpu_model_runner.py'),
    (
        '                        # Create dummy batch of multimodal inputs.\n'
        '                        batched_dummy_mm_inputs = self._get_mm_dummy_batch(\n'
        '                            dummy_modality,\n'
        '                            max_mm_items_per_batch,\n'
        '                        )\n'
        '\n'
        '                        # Run multimodal encoder.\n'
        '                        dummy_encoder_outputs = self.model.embed_multimodal(\n'
        '                            **batched_dummy_mm_inputs\n'
        '                        )\n'
        '\n'
        '                        sanity_check_mm_encoder_outputs(\n'
        '                            dummy_encoder_outputs,\n'
        '                            expected_num_items=max_mm_items_per_batch,\n'
        '                        )\n'
        '                        for i, output in enumerate(dummy_encoder_outputs):\n'
        '                            self.encoder_cache[f"tmp_{i}"] = output'
    ),
    (
        '                        # Create dummy batch of multimodal inputs.\n'
        '                        try:\n'
        '                            batched_dummy_mm_inputs = self._get_mm_dummy_batch(\n'
        '                                dummy_modality,\n'
        '                                max_mm_items_per_batch,\n'
        '                            )\n'
        '                        except Exception as _gmr_exc:\n'
        '                            import logging as _l\n'
        '                            _l.getLogger(__name__).warning(\n'
        '                                "Skipping encoder profiling for %s - "\n'
        '                                "_get_mm_dummy_batch failed (%s). "\n'
        '                                "First real request will warm up the encoder.",\n'
        '                                dummy_modality, _gmr_exc,\n'
        '                            )\n'
        '                        else:\n'
        '                            dummy_encoder_outputs = self.model.embed_multimodal(\n'
        '                                **batched_dummy_mm_inputs\n'
        '                            )\n'
        '                            sanity_check_mm_encoder_outputs(\n'
        '                                dummy_encoder_outputs,\n'
        '                                expected_num_items=max_mm_items_per_batch,\n'
        '                            )\n'
        '                            for i, output in enumerate(dummy_encoder_outputs):\n'
        '                                self.encoder_cache[f"tmp_{i}"] = output'
    ),
    'gpu_model_runner.py',
)

# ── Patch 4: nano_nemotron_vl.py ─────────────────────────────────────────────
# During dummy-input profiling, video_num_patches is empty but
# get_video_replacement is still called with item_idx=0, causing IndexError.
_patch(
    _find('nano_nemotron_vl.py'),
    '            num_patches = video_num_patches[item_idx]\n',
    '            num_patches = video_num_patches[item_idx] if item_idx < len(video_num_patches) else None\n',
    'nano_nemotron_vl.py video_num_patches bounds check',
)

# ── Patch 5: nano_nemotron_vl.py EVS pruning None guard ──────────────────────
# When num_tubelets is None (dummy profiling with empty video_num_patches),
# the EVS pruning block does `int * None` → TypeError.  Guard the condition
# and add a fallback else branch using frame count from metadata.
_nano = _find('nano_nemotron_vl.py')
_patch(
    _nano,
    (
        '            if video_pruning_rate is not None and video_pruning_rate > 0.0:\n'
        '                # Start of EVS-specific code\n'
        '                num_tokens = compute_retained_tokens_count(\n'
        '                    tokens_per_frame=feature_size,\n'
        '                    num_frames=num_tubelets,\n'
        '                    q=video_pruning_rate,\n'
        '                )\n'
        '                # Here we just need placeholders that won\'t actually be replaced -\n'
        '                # we just need to make sure the total number of tokens is correct\n'
        '                # assign all tokens to the first frame\n'
        '                tokens_per_frame = [num_tokens] + [0] * (num_tubelets - 1)\n'
        '\n'
        '                # End of EVS-specific code\n'
        '            else:\n'
        '                tokens_per_frame = [feature_size] * num_tubelets'
    ),
    (
        '            if video_pruning_rate is not None and video_pruning_rate > 0.0 and num_tubelets is not None:\n'
        '                # Start of EVS-specific code\n'
        '                num_tokens = compute_retained_tokens_count(\n'
        '                    tokens_per_frame=feature_size,\n'
        '                    num_frames=num_tubelets,\n'
        '                    q=video_pruning_rate,\n'
        '                )\n'
        '                # Here we just need placeholders that won\'t actually be replaced -\n'
        '                # we just need to make sure the total number of tokens is correct\n'
        '                # assign all tokens to the first frame\n'
        '                tokens_per_frame = [num_tokens] + [0] * (num_tubelets - 1)\n'
        '\n'
        '                # End of EVS-specific code\n'
        '            elif num_tubelets is not None:\n'
        '                tokens_per_frame = [feature_size] * num_tubelets\n'
        '            else:\n'
        '                # num_tubelets is None (dummy profiling) — use frame count as fallback\n'
        '                tokens_per_frame = [feature_size] * len(metadata.get("frames_indices", [None]))'
    ),
    'nano_nemotron_vl.py EVS pruning None guard',
)

# ── Patch 6: chat_completion/protocol.py ─────────────────────────────────────
# set_include_reasoning_for_none_effort is a Pydantic model_validator(mode="before")
# that calls data.get("reasoning_effort").  When the request body contains a
# list-typed "content" field, Pydantic can call the validator with a list, causing
# AttributeError: 'list' object has no attribute 'get'.
# Fix: bail out early when data is not a dict.
_chat_proto = None
for _r in ROOTS:
    for _p in glob.glob(_r + '/**/chat_completion/protocol.py', recursive=True):
        _chat_proto = _p
        break
_patch(
    _chat_proto,
    (
        'def set_include_reasoning_for_none_effort(cls, data: Any) -> Any:\n'
        '        if data.get("reasoning_effort") == "none":\n'
    ),
    (
        'def set_include_reasoning_for_none_effort(cls, data: Any) -> Any:\n'
        '        if not isinstance(data, dict):\n'
        '            return data\n'
        '        if data.get("reasoning_effort") == "none":\n'
    ),
    'chat_completion/protocol.py set_include_reasoning_for_none_effort',
)

print('vLLM patch complete', flush=True)
