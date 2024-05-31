def count_parameters(model):
    param_cnt = sum(p.numel() for p in model.parameters())
    param_cnt = str(param_cnt)
    cnt_n = len(param_cnt)
    param_list = [param_cnt[max(0, p-3):p]for p in range(cnt_n, 0, -3)]
    param_str = ','.join(param_list[::-1])
    return param_str