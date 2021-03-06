from agents.DCOACH import DCOACH
from agents.HG_DAgger import HG_DAGGER
"""
Functions that selects the agent
"""


def agent_selector(agent_type, config_agent):
    if agent_type == 'DCOACH':
        return DCOACH(dim_a=config_agent.getint('dim_a'),
                      dim_o=config_agent.getint('dim_o'),
                      action_upper_limits=config_agent['action_upper_limits'],
                      action_lower_limits=config_agent['action_lower_limits'],
                      buffer_min_size=config_agent.getint('buffer_min_size'),
                      buffer_sampling_rate=config_agent.getint('buffer_sampling_rate'),
                      buffer_sampling_size=config_agent.getint('buffer_sampling_size'),
                      train_end_episode=config_agent.getboolean('train_end_episode'),
                      policy_model_learning_rate=config_agent.getfloat('policy_model_learning_rate'),
                      human_model_learning_rate=config_agent.getfloat('human_model_learning_rate'),
                      agent_with_hm_learning_rate=config_agent.getfloat('agent_with_hm_learning_rate'),
                      human_model_included=config_agent.getboolean('human_model_included'),
                      e = config_agent['e'],
                      action_limit = config_agent['action_limit'],
                      buffer_max_size = config_agent.getint('buffer_max_size'),
                      h_threshold = config_agent['h_threshold']
                      )

    elif agent_type == 'HG_DAgger':
        return HG_DAGGER(dim_a=config_agent.getint('dim_a'),
                         action_upper_limits=config_agent['action_upper_limits'],
                         action_lower_limits=config_agent['action_lower_limits'],
                         buffer_min_size=config_agent.getint('buffer_min_size'),
                         buffer_max_size=config_agent.getint('buffer_max_size'),
                         buffer_sampling_rate=config_agent.getint('buffer_sampling_rate'),
                         buffer_sampling_size=config_agent.getint('buffer_sampling_size'),
                         number_training_iterations = config_agent.getint('number_training_iterations'),
                         train_end_episode=config_agent.getboolean('train_end_episode'))
    else:
        raise NameError('Not valid network.')
