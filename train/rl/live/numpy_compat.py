def patch_numpy_bitgenerator_compat():
    try:
        import numpy.random._pickle as np_pickle
    except Exception:
        return

    original_ctor = getattr(np_pickle, "__bit_generator_ctor", None)
    if original_ctor is None:
        return
    if getattr(original_ctor, "__name__", "") == "_compat_bit_generator_ctor":
        return

    tolerant_cache = {}

    def _normalize_bg_name(value):
        if isinstance(value, type):
            return value.__name__
        if isinstance(value, str):
            if "PCG64DXSM" in value:
                return "PCG64DXSM"
            if "PCG64" in value:
                return "PCG64"
            if "MT19937" in value:
                return "MT19937"
            if "Philox" in value:
                return "Philox"
            if "SFC64" in value:
                return "SFC64"
            return value
        return str(value)

    def _build_tolerant_bitgen(base_cls):
        cached = tolerant_cache.get(base_cls)
        if cached is not None:
            return cached

        class _TolerantBitGen(base_cls):
            def __setstate__(self, state):
                try:
                    super().__setstate__(state)
                    return
                except Exception:
                    pass

                if isinstance(state, tuple):
                    for candidate in state:
                        if isinstance(candidate, dict):
                            try:
                                super().__setstate__(candidate)
                                return
                            except Exception:
                                continue
                return

        _TolerantBitGen.__name__ = f"Compat{base_cls.__name__}"
        tolerant_cache[base_cls] = _TolerantBitGen
        return _TolerantBitGen

    def _compat_bit_generator_ctor(bit_generator_name="MT19937"):
        normalized = _normalize_bg_name(bit_generator_name)
        base_cls = None
        if isinstance(bit_generator_name, type):
            base_cls = bit_generator_name
        elif hasattr(np_pickle, "BitGenerators") and normalized in np_pickle.BitGenerators:
            base_cls = np_pickle.BitGenerators[normalized]

        if base_cls is None:
            return original_ctor(normalized)

        tolerant_cls = _build_tolerant_bitgen(base_cls)
        return tolerant_cls()

    np_pickle.__bit_generator_ctor = _compat_bit_generator_ctor
