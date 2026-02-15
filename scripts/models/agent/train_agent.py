# 1. Gather random context
# 2. Encode observations
# 3. Tokenize latent and actions
# 4. Reuse inference -> dream function to gather bath of imagined data
# 5. Encode imagined frames
# 6. Tokenize imagined latent and action
# 7. Concat imagined latent and token
# 8. Using data from 7., train the agent.