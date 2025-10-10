data_generation_S0 = function(S0, env.reset = custom_cartpole.reset,
                           env.step = custom_cartpole.step, 
                           policy = custom_cartpole.random_policy,
                           max_episode_len = 100) {
  
  episode_lengths = nrow(S0)
  
  
  St = NULL
  At = NULL
  Rt = NULL
  Sn = NULL
  
  for (i in seq_len(nrow(S0))) {
    
    # env.reset() 生成初始状态，并存入 S0
    state = S0[i, ]
    
    step_num = 0
    done = FALSE
    
    while (!done) {
      
      action = policy(state)
      next_step = env.step(state, action, old_step_num = step_num, max_episode_len)
      next_state = next_step$next_state
      reward = next_step$reward
      done = next_step$done
      step_num = next_step$step_num
      
      St = rbind(St, state)
      At = c(At, action)
      Rt = c(Rt, reward)
      Sn = rbind(Sn, next_state)
      
      state = next_state
    }
    
    episode_lengths[i] = step_num
  }
  
  # 转换为 data.frame，并统一列名
  S0 = as.data.frame(S0)
  St = as.data.frame(St)
  Sn = as.data.frame(Sn)
  
  state_names = paste0("s", seq_len(ncol(St)))
  names(S0) = state_names
  names(St) = state_names
  names(Sn) = state_names  
  
  return(list(
    S0 = S0,                    # 每个 episode 的初始状态
    St = St,                    # 所有时刻的状态
    At = At,                    # 动作序列
    Rt = Rt,                    # 奖励序列
    Sn = Sn,                    # 下一个状态序列
    episode_lengths = episode_lengths
  ))
}




data_generation = function(env.reset = custom_cartpole.reset,
                           env.step = custom_cartpole.step, 
                           policy = custom_cartpole.random_policy,
                           num_episodes = 100,
                           max_episode_len = 100) {
  
  episode_lengths = integer(num_episodes)
  
  # 新增：初始化 S0 用于保存每个 episode 的初始状态
  S0 = NULL
  
  St = NULL
  At = NULL
  Rt = NULL
  Sn = NULL
  
  for (i in seq_len(num_episodes)) {
    
    # env.reset() 生成初始状态，并存入 S0
    state = env.reset()
    S0 = rbind(S0, state)
    
    step_num = 0
    done = FALSE
    
    while (!done) {
      
      action = policy(state)
      next_step = env.step(state, action, old_step_num = step_num, max_episode_len)
      next_state = next_step$next_state
      reward = next_step$reward
      done = next_step$done
      step_num = next_step$step_num
      
      St = rbind(St, state)
      At = c(At, action)
      Rt = c(Rt, reward)
      Sn = rbind(Sn, next_state)
      
      state = next_state
    }
    
    episode_lengths[i] = step_num
  }
  
  # 转换为 data.frame，并统一列名
  S0 = as.data.frame(S0)
  St = as.data.frame(St)
  Sn = as.data.frame(Sn)
  
  state_names = paste0("s", seq_len(ncol(St)))
  names(S0) = state_names
  names(St) = state_names
  names(Sn) = state_names  
  
  return(list(
    S0 = S0,                    # 每个 episode 的初始状态
    St = St,                    # 所有时刻的状态
    At = At,                    # 动作序列
    Rt = Rt,                    # 奖励序列
    Sn = Sn,                    # 下一个状态序列
    episode_lengths = episode_lengths
  ))
}






empirical_returns = function(env.reset = custom_cartpole.reset, env.step = custom_cartpole.step, 
                             policy, disc_fact = 0.95, num_episodes = 100, max_episode_len = 100, force_magnitude = 10.0,
                             mass_cart = 1.0,mass_pole = 0.1, pole_length = 0.5){
  
  
  returns = rep(0, num_episodes)
  for (i in 1:num_episodes) {
    
    gamma = 1
    one_return = 0
    
    state = env.reset()
    step_num = 0
    done = FALSE
    
    while (!done) {
      
      action = policy(state)
      next_step = env.step(state, action, old_step_num = step_num, max_episode_len, force_magnitude = force_magnitude,
                           mass_cart = mass_cart, mass_pole = mass_pole, pole_length = pole_length)
      next_state = next_step$next_state
      reward = next_step$reward
      done = next_step$done
      step_num = next_step$step_num
      
      
      state = next_state
      
      one_return = one_return + reward*gamma
      gamma = gamma*disc_fact
      
    }
    
    returns[i] = one_return
    
  }  
  
  return(mean(returns))
  
}







true_return_simu = function(S0, env.reset = custom_cartpole.reset, env.step = custom_cartpole.step, 
                             policy, disc_fact = 0.95, num_episodes = 100, max_episode_len = 100, force_magnitude = 10.0,
                             mass_cart = 1.0,mass_pole = 0.1, pole_length = 0.5){
  
  
  returns = matrix(0,nrow = nrow(S0), ncol = num_episodes)
  
  for (j in 1:nrow(S0)) {
    
    for (i in 1:num_episodes) {
      
      gamma = 1
      one_return = 0
      
      state = S0[j, ]
      step_num = 0
      done = FALSE
      
      while (!done) {
        
        action = policy(state)
        next_step = env.step(state, action, old_step_num = step_num, max_episode_len, force_magnitude = force_magnitude,
                             mass_cart = mass_cart,mass_pole = mass_pole, pole_length = pole_length)
        next_state = next_step$next_state
        reward = next_step$reward
        done = next_step$done
        step_num = next_step$step_num
        
        
        state = next_state
        
        one_return = one_return + reward*gamma
        gamma = gamma*disc_fact
        
      }
      
      returns[j, i] = one_return
      
    }
    
    if(j%%50 == 0) print(j)
    
  }  
  
  return(rowMeans(returns))
  
}























