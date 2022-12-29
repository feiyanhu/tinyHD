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

#default decoder inflate configs!!!
config2 = [[((),()), ((3, 2, 1),()), (), ()]]
config1 = [[((),()), ((1, 1, 0),()), (), ()]]
decoder_inflate_config = [
    config2, config1, config2, config2,
    config2, config1, config2, config1,
    config2, config1, config1, config1,
    config2, config1, config1, config1,
    #second decoder
    config1, config1, config2, config1, config2, config2, config2, config2,
    #third decoder
    config2, config2, config1, config1, config1, config2, config2, config2, config2, config2, config2, config2, config1,
]

#(28, 6), (24, 34), (136, 130), 
# (6, 8), (34, 32), (130, 128), (7, 2), (6, 8), (34, 32),
              #(2, 2), (8, 8), (2, 2), (2, 1),