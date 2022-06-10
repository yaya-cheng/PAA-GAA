import torch

@torch.no_grad()
def test(x_adv, y, target_y, whitebox, blackbox1, blackbox2, blackbox3, list_white, list_y, num, utr, tsuc, ttr):
    x_adv = x_adv.cuda()
    y = y.cuda()
    pred_adv_b1 = torch.argmax(blackbox1(x_adv), dim=1)
    pred_adv_b2 = torch.argmax(blackbox2(x_adv), dim=1)
    pred_adv_b3 = torch.argmax(blackbox3(x_adv), dim=1)
    pred_adv_w = torch.argmax(whitebox(x_adv), dim=1)

    # White Box Model
    num[0] += torch.sum(pred_adv_w != y)
    tsuc[0] += torch.sum(pred_adv_w == target_y)
    idx_w = pred_adv_w != y
    idx_w_t = pred_adv_w == target_y
    # Save White Box Model Tsuc List
    for img in x_adv[idx_w_t]:
        list_white.append(img.detach().cpu().numpy())
    for t_y in target_y[idx_w_t]:
        list_y.append(t_y.detach().cpu().numpy())
    
    # Black Box Model
    num[1] += torch.sum(pred_adv_b1  != y)
    tsuc[1] += torch.sum(pred_adv_b1  == target_y)
    idx_b1 = pred_adv_b1 != y
    idx_b1_t = pred_adv_b1 == target_y
    num[2] += torch.sum(pred_adv_b2 != y)
    tsuc[2] += torch.sum(pred_adv_b2 == target_y)
    idx_b2 = pred_adv_b2 != y
    idx_b2_t = pred_adv_b2 == target_y
    num[3] += torch.sum(pred_adv_b3 != y)
    tsuc[3] += torch.sum(pred_adv_b3 == target_y)
    idx_b3 = pred_adv_b3 != y
    idx_b3_t = pred_adv_b3 == target_y
    utr[0] += torch.sum(idx_w & idx_b1)
    utr[1] += torch.sum(idx_w & idx_b2)
    utr[2] += torch.sum(idx_w & idx_b3)
    ttr[0] += torch.sum(idx_w_t & idx_b1_t)
    ttr[1] += torch.sum(idx_w_t & idx_b2_t)
    ttr[2] += torch.sum(idx_w_t & idx_b3_t)

    return list_white, list_y, num, utr, tsuc, ttr 