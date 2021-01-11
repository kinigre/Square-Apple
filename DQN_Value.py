class Value:
    EXPLORE = 4000000 #explore
    N_ACTIONS = 4 #n_actions
    LEARNING_RATE = 0.001 #learning_rate
    GAMMA = 0.9 # 한 번에 볼 총 프레임 수 입니다.
    REPLACE_TARGET_ITER = 2000 # replace_target_iter
    MEMORY_SIZE = 10000 #학습에 사용할 플레이결과를 얼마나 많이 저장해서 사용할지를 정합니다.
    BATCH_SIZE = 256 # 과거의 상태에 대한 가중치를 줄이는 역할을 합니다.
    FINAL_EPSILON = 0.001 #final_epsilon
    INITIAL_EPSILON = 0.001 #initial_epsilon
    OBSERVE = 1000 #observe
    MODEL_FILE = './model/snake' #model_file