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

# ── Patch 7: multimodal/processing/processor.py _merge_mm_kwargs ─────────────
# Two related IndexError sites in _merge_mm_kwargs:
#
# Case A — thumbnail/tile mismatch: NanoNemotronVL produces 2 hash entries
#   (thumbnail + tile) but mm_missing_kwargs has only 1 item (the combined
#   pixel_values tensor).  Iteration 2 does missing_kwargs[1] → IndexError.
#
# Case B — empty modality list: for image-only requests the video modality
#   still gets hash entries in mm_hashes but zero kwargs items, so
#   missing_kwargs == [] and min(x, -1) == -1 → list[-1] on empty → IndexError.
#
# Fix: wrap the block in `if missing_kwargs:`.  When non-empty, clamp the index
# to the last item (handles Case A).  When empty, fall through to item=None so
# cache.get_and_update_item treats it as a cache-only lookup (handles Case B).
_mmproc = None
for _r in ROOTS:
    for _p in glob.glob(_r + '/**/multimodal/processing/processor.py', recursive=True):
        _mmproc = _p
        break
_patch(
    _mmproc,
    (
        '                    missing_kwargs_item = missing_kwargs[missing_next_idx]\n'
        '                    missing_updates_item = missing_prompt_updates[missing_next_idx]\n'
        '\n'
        '                    mm_missing_next_idx[modality] += 1\n'
        '\n'
        '                    item = missing_kwargs_item, missing_updates_item\n'
    ),
    (
        '                    if missing_kwargs:\n'
        '                        # Guard: NanoNemotronVL thumbnail/tile mismatch and\n'
        '                        # empty-modality case in vLLM 0.19.0. Clamp index to\n'
        '                        # last available item; skip entirely when list is empty.\n'
        '                        _mi = min(missing_next_idx, len(missing_kwargs) - 1)\n'
        '                        _pi = min(missing_next_idx, len(missing_prompt_updates) - 1)\n'
        '                        missing_kwargs_item = missing_kwargs[_mi]\n'
        '                        missing_updates_item = missing_prompt_updates[_pi]\n'
        '                        mm_missing_next_idx[modality] += 1\n'
        '                        item = missing_kwargs_item, missing_updates_item\n'
        '                    else:\n'
        '                        # No kwargs items for this modality — treat as\n'
        '                        # cache-only lookup (image/video mismatch).\n'
        '                        item = None\n'
    ),
    'multimodal/processing/processor.py _merge_mm_kwargs bounds check',
)

# ── Patch 8: NanoNemotronVLMultiModalProcessor._call_hf_processor ────────────
# Root cause of the "found 0 image items" RuntimeError on every inference:
#
# NanoNemotronVLProcessor does not expose image_processor / video_processor /
# feature_extractor as sub-attributes.  call_hf_processor_mm_only() (used by
# _apply_hf_processor_mm_only when no _call_hf_processor override exists) relies
# on those attributes to call each sub-processor separately; when none are found
# it returns an empty BatchFeature.  _apply_hf_processor_main then feeds this
# empty result to from_hf_inputs, producing 0 kwargs items, and
# _validate_mm_kwargs raises:
#     RuntimeError: Expected there to be 1 image items … but only found 0!
#
# Fix: add _call_hf_processor to NanoNemotronVLMultiModalProcessor.  It is
# intentionally identical to the base-class implementation.  The only effect is
# that  type(self)._call_hf_processor != BaseMultiModalProcessor._call_hf_processor
# becomes True, which causes _apply_hf_processor_mm_only to take the
# dummy-text branch — it generates a placeholder prompt and calls the FULL
# NanoNemotronVLProcessor.__call__ (which handles images and videos natively)
# instead of the broken call_hf_processor_mm_only fallback.
_nano_model = None
for _r in ROOTS:
    for _p in glob.glob(_r + '/**/model_executor/models/nano_nemotron_vl.py', recursive=True):
        _nano_model = _p
        break
_patch(
    _nano_model,
    (
        'class NanoNemotronVLMultiModalProcessor(\n'
        '    BaseMultiModalProcessor[NanoNemotronVLProcessingInfo]\n'
        '):\n'
        '    def _get_image_fields_config(self, hf_inputs: BatchFeature):\n'
    ),
    (
        'class NanoNemotronVLMultiModalProcessor(\n'
        '    BaseMultiModalProcessor[NanoNemotronVLProcessingInfo]\n'
        '):\n'
        '    def _call_hf_processor(\n'
        '        self,\n'
        '        prompt: str,\n'
        '        mm_data: "Mapping[str, object]",\n'
        '        mm_kwargs: "Mapping[str, object]",\n'
        '        tok_kwargs: "Mapping[str, object]",\n'
        '    ) -> "BatchFeature":\n'
        '        # Overriding this method (even with the same body as the base class)\n'
        '        # causes _apply_hf_processor_mm_only to use the dummy-text path\n'
        '        # rather than call_hf_processor_mm_only.  The latter tries\n'
        '        # processor.image_processor / .video_processor which\n'
        '        # NanoNemotronVLProcessor does not expose, returning an empty\n'
        '        # BatchFeature and making image inference fail with\n'
        '        # "found 0 image items".\n'
        '        return self.info.ctx.call_hf_processor(\n'
        '            self.info.get_hf_processor(**mm_kwargs),\n'
        '            dict(text=prompt, **mm_data),\n'
        '            dict(**mm_kwargs, **tok_kwargs),\n'
        '        )\n'
        '\n'
        '    def _get_image_fields_config(self, hf_inputs: BatchFeature):\n'
    ),
    'nano_nemotron_vl.py NanoNemotronVLMultiModalProcessor._call_hf_processor',
)

print('vLLM patch complete', flush=True)
