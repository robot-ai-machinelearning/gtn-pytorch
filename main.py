import random
from gtn.config.config import LearnerConfig

random.seed(1)
learner_config = LearnerConfig(random.randint(32, 128), random.randint(64, 256), random.randint(64, 256))
print(learner_config.learner_conv1_filters)