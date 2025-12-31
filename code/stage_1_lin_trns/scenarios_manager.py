
def generate_default_scenarios():
    """Generates a standard suite of stability tests."""
    scenarios = []
    
    # 1. Rotation Sweep
    for angle in [30, 45, 90, 180]:
        scenarios.append({
            'name': f'Rotate {angle}°',
            'angle': angle, 'scaling': 1.0, 'translation': 0, 'flip': False
        })
        
    # 2. Scaling
    scenarios.append({'name': 'Scale x0.8', 'angle': 0, 'scaling': 0.8, 'translation': 0, 'flip': False})
    scenarios.append({'name': 'Scale x1.2', 'angle': 0, 'scaling': 1.2, 'translation': 0, 'flip': False})
    
    # 3. Translation (Shift)
    scenarios.append({'name': 'Shift X+10', 'angle': 0, 'scaling': 1.0, 'translation': (10, 0), 'flip': False})
    
    # 4. Flip
    scenarios.append({'name': 'Flip X', 'angle': 0, 'scaling': 1.0, 'translation': 0, 'flip': True})
    
    return scenarios



def check_scenario_cache(methods_to_test, config, former_results):
    """
    Checks if results for the given scenario and methods already exist in the cache.
    Returns:
        methods_to_run: List of methods that are missing and need computation.
    """
    methods_to_run = []
    
    # Extract scenario params for filtering
    name = config.get('name')
    angle = config.get('angle')
    scaling = config.get('scaling')
    # Handle translation being a tuple/list which needs string comparison in pandas
    trans_val = str(config.get('translation')) 
    flip = config.get('flip')

    for method in methods_to_test:
        found_in_cache = False
        
        if not former_results.empty:
            match = former_results[
                (former_results['method'] == method) &
                (former_results['name'] == name) &
                (former_results['angle'] == angle) &
                (former_results['scaling'] == scaling) &
                (former_results['translation'].astype(str) == trans_val) &
                (former_results['flip'] == flip)
            ]
            
            if not match.empty:
                print(f"  -> [{method}] Using cached results.")
                found_in_cache = True
        
        if not found_in_cache:
            methods_to_run.append(method)
            
    return methods_to_run