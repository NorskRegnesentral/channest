def progress(itr, *args, **kwargs):
    try:
        import tqdm
        return tqdm.tqdm(itr, *args, **kwargs)
    except IndexError:
        return itr
