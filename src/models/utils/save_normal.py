    # """ Save normal and mask"""

    #         # temp = nout[0, :, :, :].permute(1,2,0).cpu().detach().numpy()
    #         # ### swap for surface normal integration
    #         # normal_map_est = np.zeros(temp.shape, np.float32)
    #         # normal_map_est[:, :, 0] = temp[:, :, 1]
    #         # normal_map_est[:, :, 1] = temp[:, :, 0]
    #         # normal_map_est[:, :, 2] = -temp[:, :, 2]
    #         # temp = nml[0, :, :, :].permute(1,2,0).cpu().detach().numpy()
    #         # normal_map_gt = np.zeros(temp.shape, np.float32)
    #         # normal_map_gt[:, :, 0] = temp[:, :, 1]
    #         # normal_map_gt[:, :, 1] = temp[:, :, 0]
    #         # normal_map_gt[:, :, 2] = -temp[:, :, 2]
    #         # mask = m[0, 0, :, :].cpu().detach().numpy().astype(np.bool)
    #         # result={'normal_map_est':normal_map_est, 'normal_map_gt': normal_map_gt ,'mask':mask}
    #         loss = 0
    #         return loss, output.cpu().detach().numpy(), input.cpu().detach().numpy(), result