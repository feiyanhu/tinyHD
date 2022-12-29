def all_config():
    #model_path = '../pretrained/d1d2d3_M_lt.pth'
    ###TABLE 3
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d1s/ft_199_0.33540_1.51855.pth' #[0.89930011 0.71382875 0.39390699]
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d2s/ft_191_0.38075_1.49977.pth' #[0.90396785 0.70832649 0.38203093]
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d3s/ft_190_0.38281_1.51521.pth'

    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d1s_d1s/ft_198_0.33885_1.51840.pth' #[0.89979542 0.71382004 0.39282847]
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d2s_d2s/ft_197_0.38650_1.49708.pth'
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d3s_d3s/ft_197_0.37937_1.49973.pth'

    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d1s_d1s_d1s/ft_150_0.34033_1.49460.pth' #0.90131676 0.82526991 0.49224418 2.84195852 0.392363
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d2s_d2s_d2s/ft_169_0.38436_1.51231.pth' #0.90491552 0.82656973 0.48474312 2.80423363 0.37741301
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d3s_d3s_d3s/ft_160_0.38480_1.49456.pth' #0.90473397 0.82424317 0.4845227  2.79665658 0.37992114

    #model_path = '../pretrained/d1d2d3_M_lt.pth'

    ####TABLE 4
    #model_path = '../dhf1k_l_myschedule_e1_d123s/ft_150_0.00000_1.47729.pth' #??? [0.90290433 0.70071454 0.37866744]
    #model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc_e1d3_noaug_lonly/ft_185_0.00000_1.46667.pth' #[0.90104332 0.68931009 2.73302048 0.38187308]
    #model_path = '../dhf1k_kinetic_v4_s3d_label_oldschedule_rc_e1d3_noaug_lonly/ft_180_0.00000_1.48809.pth' #[0.90316664 0.6968031  2.76730137 0.376393  ] mixed
    #labserver
    model_path = '../dhf1k_lt_myschedule_e1_d123s/ft_199_0.37454_1.48797.pth'
    model_path = '../dhf1k_t_myschedule_e1_d123s/ft_183_0.29523_0.00000.pth'
    model_path = '../kinetic_t_myschedule_e1_d123s/ft_199_0.25959_0.00000.pth' #[0.89796175 0.36566851] ?? [0.89792909 0.71137693 0.36566851]
    model_path = '../dhf1k_kinetic_t_myschedule_e1_d123s/ft_190_0.25744_0.00000.pth'

    ####TABLE 5
    model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc2_e1d3_noaug/ft_199_0.40973_1.49512.pth' #[0.90381259 0.70626649 0.36407852]
    model_path = '../dhf1k_kinetic_lt_myschedule_e1_d123s_rc2_rc1T/ft_170_0.13537_1.46981.pth' #labserver 0.90521812 0.83303276 0.48047339 2.73167188 0.36842074
    
    model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc4_e1d3_noaug/ft_198_0.44849_1.56443.pth'
    model_path = '../dhf1k_kinetic_lt_myschedule_e1_d123s_rc4_rc1T/ft_145_0.19246_1.51595.pth' #labserver 0.90178638 0.83179891 0.46673366 2.6328899  0.35691549

    #model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc2_e1d3_3dd/ft_192_0.44286_1.54738.pth'
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d123m_rc2_rc1T/ft_154_0.18415_1.52582.pth' #0.90207978 0.83072765 0.47175678 2.67264573 0.36297555
    
    #model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc4_e1d3_3dd/ft_196_0.49398_1.58137.pth'
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d123m_rc4_rc1T/ft_128_0.22907_1.55651.pth' #0.89991784 0.83325807 0.45640186 2.55812889 0.34780966


def supp_config():
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d1m/ft_178_0.37466_1.56646.pth'

    model_path = '../dhf1k_kinetic_lt_myschedule_e1_d1sf/ft_186_0.39853_1.55912.pth'
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d2m/ft_193_0.44210_1.51686.pth'
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d3m/ft_199_0.42919_1.52106.pth'

    model_path = '../dhf1k_kinetic_lt_myschedule_e1_d1sf_d1sf/ft_197_0.39860_1.54491.pth'
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d2m_d2m/ft_136_0.43204_1.52385.pth'
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d3m_d3m/ft_199_0.43429_1.53756.pth'

    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d1sf_d1sf_d1sf/ft_189_0.38811_1.53651.pth'
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d2m_d2m_d2m/ft_186_0.43762_1.51910.pth' #0.90522346 0.82586126 0.4841526  2.77675994 0.37733681
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d3m_d3m_d3m/ft_164_0.43230_1.52388.pth' #0.90280562 0.82208363 0.47685767 2.7344885  0.37562613

    #model_path = '../pretrained/d1d2d3_M_lt.pth'

    #16 in 8 out
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d1s_d2m_d3m_8out/ft_123_0.36822_1.48259.pth' #0.9073162  0.82587103 0.49614876 2.86858675 0.39135761

    #vinet and tased
    #model_path = '/home/feiyan/runs/dhf1k_kinetic_lt_myschedule_e1_d123s_vinet/ft_196_0.31956_1.42004.pth' #0.90728119 0.84998196 0.49364658 2.76337746 0.36668876
    #model_path = '/home/feiyan/runs/dhf1k_kinetic_lt_myschedule_e1_d123s_tased/ft_185_0.38896_1.42080.pth' #0.90809791 0.85194502 0.49742036 2.77508049 0.37052805


if __name__ == '__main__':
    all_config()