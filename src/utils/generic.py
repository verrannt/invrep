def progress_bar(fraction:float, bar_len:int=25):
    ''' Display a progress bar given a fraction
    between 0. and 1.
    
    Example:
    [In] progress_bar(0.4)
    [Out] '[==========>              ]'
    
    Params
    ===
    fraction : the progress as fraction between 0. and 1.
    bar_len : the length of the bar
    
    Returns
    ===
    bar : the progress bar as string of length bar_len+2
    '''
    if fraction < 0. or fraction > 1.:
        raise ValueError('fraction must be between 0. and 1.')
    
    n_done = int(fraction*bar_len)
    n_left = bar_len - n_done - 1

    bar = '['
    for _ in range(n_done):
        bar += '='
    if fraction < 1.:
        bar += '>'
    for _ in range(n_left):
        bar += ' '
    bar += ']'
    return bar

