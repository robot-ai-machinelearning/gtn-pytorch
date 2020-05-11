import random
from gtn.config.config import LearnerConfig, GeneratorConfig
from gtn.models import Learner, Generator

random.seed(1)
LearnerConfig(random.randint(32, 128), random.randint(64, 256), random.randint(64, 256), 10)
generator = Generator(GeneratorConfig)
print(generator)
for i in range(3):

    learner = Learner(LearnerConfig(random.randint(32, 128), random.randint(64, 256), random.randint(64, 256), 10))
    print(learner)