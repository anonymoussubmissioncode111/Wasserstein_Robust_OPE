custom_cartpole.reset = function(){
  
  
  return(runif(4, -0.05, 0.05))
  
}




custom_cartpole.step = function(state, action, old_step_num, max_episode_len, dt = 0.02, force_magnitude = 10.0, gravity = 9.8, 
                                mass_cart = 1.0, mass_pole = 0.1, pole_length = 0.5) {
  # Unpack the state vector
  x <- state[1]  # Cart position
  x_dot <- state[2]  # Cart velocity
  theta <- state[3]  # Pole angle
  theta_dot <- state[4]  # Pole angular velocity
  
  # Total mass and intermediate values
  total_mass <- mass_cart + mass_pole
  pole_mass_length <- mass_pole * pole_length
  
  # Force based on action
  force <- ifelse(action == 1, force_magnitude, -force_magnitude)
  
  # Sine and cosine of the pole angle
  sin_theta <- sin(theta)
  cos_theta <- cos(theta)
  
  # Calculate angular acceleration (ddot_theta)
  temp = (force + pole_mass_length * theta_dot^2 * sin_theta) / total_mass
  theta_acc = (gravity * sin_theta - cos_theta * temp) / 
    (pole_length * (4.0 / 3.0 - (mass_pole * cos_theta^2) / total_mass))
  
  # Calculate cart acceleration (ddot_x)
  x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass
  
  # Update the state using Euler's method
  x_next <- x + x_dot * dt
  x_dot_next <- x_dot + x_acc * dt
  theta_next <- theta + theta_dot * dt
  theta_dot_next <- theta_dot + theta_acc * dt
  
  #next state as a vector
  next_state = c(x_next, x_dot_next, theta_next, theta_dot_next)
  
  
  done = (abs(next_state[1])>2.4 | abs(next_state[3])> pi/15)
  
  reward = (2 - abs(next_state[1]/2.4))*(2 - abs(next_state[3]/(pi/15))) - 1
  
  step_num = old_step_num + 1
  
  if(step_num >= max_episode_len) done = TRUE
  
  return(list(next_state = next_state, reward = reward, done = done, step_num = step_num))
  
}









custom_cartpole.random_policy = function(state){
  
  return(sample(0:1, size = 1))
}