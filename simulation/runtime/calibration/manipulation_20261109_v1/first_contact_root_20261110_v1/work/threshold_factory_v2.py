import math

def build_factory(p, world_cls, info_list):
    def factory(cfg, case, q5, g):
        raw = cfg["diagnostic_threshold"]
        if isinstance(raw, bool):
            raise ValueError("diagnostic_threshold must not be bool")
        threshold = float(raw)
        if not math.isfinite(threshold) or threshold <= 0 or threshold > 0.02:
            raise ValueError("diagnostic_threshold must be finite, positive, and at most 0.02")

        forwarded_updates = []
        constructor_completed = False

        original_setter = p.setPhysicsEngineParameter

        def wrapped_setter(*args, **kwargs):
            if 'contactBreakingThreshold' in kwargs:
                new_kwargs = dict(kwargs)
                old_value = new_kwargs['contactBreakingThreshold']
                new_kwargs['contactBreakingThreshold'] = threshold
                forwarded_updates.append({
                    'original': old_value,
                    'requested': threshold,
                    'physicsClientId': new_kwargs.get('physicsClientId'),
                })
                return original_setter(*args, **new_kwargs)
            return original_setter(*args, **kwargs)

        p.setPhysicsEngineParameter = wrapped_setter
        try:
            world = world_cls(cfg, case, q5, g)
            constructor_completed = True
            return world
        finally:
            p.setPhysicsEngineParameter = original_setter
            info_list.append({
                'threshold': threshold,
                'forwarded_updates': forwarded_updates,
                'constructor_completed': constructor_completed,
            })

    return factory
