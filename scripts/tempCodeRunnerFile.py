elif torch.backends.mps.is_available(): # MPS support can be less stable
    #     device = torch.device('mps')