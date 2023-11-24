def ip_2_subnet(ip: str, net_mask: int):
    mask_zero = 32 - int(net_mask)
    count = -1
    ip_split = ip.split('.')
    while mask_zero >= 1:
        if mask_zero > 8:
            mask_zero_turn = 8
        else:
            mask_zero_turn = mask_zero
        ip_split[count] = str(int(ip_split[count]) & int(bin(0)[2:].zfill(mask_zero_turn), 2))
        mask_zero -= 8
    return ".".join(ip_split)
