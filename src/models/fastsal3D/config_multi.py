#rc 0.5 [15, 56, 8, 96, 128], [15, 48, 4, 48, 64], [15, 272, 2, 24, 32], [15, 1040, 2, 12, 16]
#rc 1   [15, 28, 8, 96, 128], [15, 24, 4, 48, 64], [15, 136, 2, 24, 32], [15, 520, 2, 12, 16]
#rc 2   [15, 14, 8, 96, 128], [15, 12, 4, 48, 64], [15, 68, 2, 24, 32],  [15, 260, 2, 12, 16]
#rc 8   [15, 8, 8, 96, 128],  [15, 12, 4, 48, 64], [15, 68, 2, 24, 32],  [15, 260, 2, 12, 16]

#encoder channel is 1
inv_config_1 = [(28, 24),  (24, 48), (12, 24), (24, 1),  (24, 56),   (14, 56), (14, 28), (28, 1),
              (136, 56), (14, 56), (14, 56), (14, 1),  (130, 136), (34, 136), (34, 136), (34, 1),
              #2nd decoder
              (520, 544),
              (136, 96), (24, 24),
              (24, 112), (28, 28), (28, 28),
              (28, 28), (7, 1),
              #3rd decoder
              (28, 6), (24, 34), (136, 130), (6, 8), (34, 32), (130, 128), (7, 2), (6, 8), (34, 32),
              (2, 2), (8, 8), (2, 2), (2, 1),
              #decoder apt
              #(28, 28), (24, 24), (136, 136), (520, 520), (28, 28), (24, 24), (136, 136), (520, 520)
             ]
inv_config_1_2 = [(42, 24),  (24, 48), (12, 24), (24, 1),  (36, 56),   (14, 56), (14, 28), (28, 1),
                  (204, 56), (14, 56), (14, 56), (14, 1),  (195, 136), (34, 136), (34, 136), (34, 1),
                  #2nd decoder
                  (780, 816),
                  (204, 144), (36, 36),
                  (36, 168), (42, 42), (42, 42),
                  (42, 40), (10, 1),
                  #3rd decoder
                  (42, 9), (36, 51), (204, 195), (9, 8), (51, 32), (195, 128), (42, 2), (9, 8), (51, 32),
                  (2, 2), (8, 8), (2, 2), (2, 1)
                 ]
#2 times less
inv_config_2 = [(14, 24),  (24, 48), (12, 24), (24, 1),  (12, 56),   (14, 56), (14, 28), (28, 1),
              (68, 56), (14, 56), (14, 56), (14, 1),  (65, 136), (34, 136), (34, 136), (34, 1),
              #2nd decoder
              (260, 272),
              (68, 48), (12, 12),
              (12, 56), (14, 14), (14, 14),
              (14, 14), (14, 1),
              #3rd decoder
              (14, 12), (12, 68), (68, 260), (12, 8), (68, 32), (260, 128), (14, 2), (12, 8), (68, 32),
              (2, 2), (8, 8), (2, 2), (2, 1)
             ]
# 4 times less
inv_config_4 = [(7, 24),  (24, 48), (12, 24), (24, 1),  (6, 56),   (14, 56), (14, 28), (28, 1),
              (34, 56), (14, 56), (14, 56), (14, 1),  (130, 136), (34, 136), (34, 136), (34, 1),
              #2nd decoder
              (130, 136),
              (34, 24), (6, 6),
              (6, 28), (7, 7), (7, 7),
              (7, 7), (7, 1),
              #3rd decoder
              (7, 6), (6, 34), (34, 130), (6, 8), (34, 32), (130, 128), (7, 2), (6, 8), (34, 32),
              (2, 2), (8, 8), (2, 2), (2, 1)
             ]
# 8 times less
inv_config_8 = [(8, 24),  (24, 48), (12, 24), (24, 1),  (12, 56),   (14, 56), (14, 28), (28, 1),
              (68, 56), (14, 56), (14, 56), (14, 1),  (65, 136), (34, 136), (34, 136), (34, 1),
              #2nd decoder
              (260, 272),
              (68, 48), (12, 12),
              (12, 32), (8, 8), (8, 8),
              (8, 8), (8, 1),
              #3rd decoder
              (8, 12), (12, 68), (68, 260), (12, 8), (68, 32), (260, 128), (8, 2), (12, 8), (68, 32),
              (2, 2), (8, 8), (2, 2), (2, 1)
             ]

#inflate mobilenet config default !!!!!
default_invert = [((),()), ((),()), (), ()]
default_invert2 = [((),()), ((2,1,0),()), (), ()]

mbv2_inflate_config = [
   ((), ()), #last stride2
   [((),()), (), ()], 
   [((),()), ((3, 1, 1),()), (), ()], #last stride2
   default_invert, default_invert, #last stride 2
   default_invert, default_invert, default_invert, #last stride 2
   default_invert, default_invert, default_invert, default_invert,
   default_invert, default_invert, [((),()), ((3, 1, 1),()), (), ()], #last stride 2
   default_invert, default_invert, default_invert,
   ((), ())
]

#3d settings
default_invert3 = [((),()), ((1,1,0),()), (), ()]
mbv2_inflate_config_3dd = mbv2_inflate_config.copy()
mbv2_inflate_config_3dd[14] = [((),()), ((2, 1, 0),()), (), ()]
mbv2_inflate_config_3dd[15:18] = [default_invert3, default_invert3, default_invert3]

config2 = [[((),()), ((3, 2, 1),()), (), ()]]
config1 = [[((),()), ((3, 1, 1),()), (), ()]] #110
config1_ = [[((),()), ((1, 1, 0),()), (), ()]]
decoder_inflate_config_3dd = [
    config2, config1, config2, config2,
    config2, config1, config2, config1,
    config2, config1, config1, config1,
    config1, config1, config1, config1,
    #second decoder
    config1, config1, config2, config1, config2, config2, config2, config2,
    #third decoder
    config2, config2, config1, config1, config1, config2, config2, config2, config2, config2, config2, config2, config1,
]

decoder_inflate_config_3dd_d1m = [
    config1, config1, config1, config1,
    config1, config1, config1, config1,
    config1, config1, config1, config1,
    config1, config1, config1, config1,
    #second decoder
    config1, config1, config1, config1, config1, config1, config1, config1,
    #third decoder
    config1, config1, config1, config1, config1, config1, config1, config1, config1, config1, config1, config1, config1,
]

decoder_inflate_config_3dd_d1m_8 = [
    config1, config1, config1, config1,
    config1, config1, config1, config1,
    config1, config1, config1, config1,
    config1, config1, config1, config1,
    #second decoder
    config1, config1, config1, config1, config1, config1, config1, config1,
    #third decoder
    config2, config2, config2, config1, config1, config1, config1, config1, config1, config1, config1, config1, config1,
]