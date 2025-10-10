toy.reset = function(){
  
  return(runif(1, 2, 3))
  
}




toy.step = function(state, action, old_step_num, max_episode_len) {
  
  
  if(action == 0) {
    next_state = runif(1, max(0, state-0.2), min(5, state+1))
  } else {
    next_state = runif(1, max(0, 0.2*state-0.02), min(5, state+0.5))
  }
  
  
  done = FALSE
  
  reward = (26 - state^2 - (action == 1))/26
  
  step_num = old_step_num + 1
  
  if(step_num >= max_episode_len) done = TRUE
  
  return(list(next_state = next_state, reward = reward, done = done, step_num = step_num))
  
}




toy.random_policy = function(state){
  
  return(sample(0:1, size = 1))
}






data_generation = function(env.reset = toy.reset,
                           env.step = toy.step, 
                           policy = toy.random_policy,
                           num_episodes = 100,
                           max_episode_len = 100) {
  
  episode_lengths = integer(num_episodes)

  S0 = NULL
  
  St = NULL
  At = NULL
  Rt = NULL
  Sn = NULL
  
  for (i in seq_len(num_episodes)) {
    
  
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
  

  S0 = as.data.frame(S0)
  St = as.data.frame(St)
  Sn = as.data.frame(Sn)
  
  state_names = paste0("s", seq_len(ncol(St)))
  names(S0) = state_names
  names(St) = state_names
  names(Sn) = state_names  
  
  return(list(
    S0 = S0,                    
    St = St,                    
    At = At,                  
    Rt = Rt,                    
    Sn = Sn,                   
    episode_lengths = episode_lengths
  ))
}







empirical_returns = function(env.reset = toy.reset, env.step = toy.step, 
                             policy = target_policy, disc_fact = 0.95, num_episodes = 1000, max_episode_len = 100){
  
  
  returns = rep(0, num_episodes)
  for (i in 1:num_episodes) {
    
    gamma = 1
    one_return = 0
    
    state = env.reset()
    step_num = 0
    done = FALSE
    
    while (!done) {
      
      action = policy(state)
      next_step = env.step(state, action, old_step_num = step_num, max_episode_len)
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










