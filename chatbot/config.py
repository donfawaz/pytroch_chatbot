class Config:
    # Configure models
    model_name = 'cb_model'
    attn_model = 'dot'
    #attn_model = 'general'
    #attn_model = 'concat'
    hidden_size = 50 # 500
    encoder_n_layers = 1 # 2
    decoder_n_layers = 1
    dropout = 0.1
    batch_size = 8 # 64
    save_dir = '.'
    corpus_name = 'Arabic_chatbot'
    datafile = '/Users/fawazalqaoud/Documents/project/python/arabic_chatbot/data/arabic_q_a.pkl'
    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token
    USE_MULTINOMIAL = False
    TEMP = 0.9
    MAX_LENGTH = 10  # Maximum sentence length to consider
    MIN_COUNT = 3    # Minimum word count threshold for trimming
    
    # Configure training/optimization
    clip = 50.0
    teacher_forcing_ratio = 0.9
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_iteration = 50
    print_every = 10
    save_every = 50