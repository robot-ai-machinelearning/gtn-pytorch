from dataclasses import dataclass

@dataclass
class LearnerConfig:
    learner_conv1_filters: int
    learner_conv2_filters: int
    learner_fc_filters: int
    learner_classes: int
    learner_leaky_relu_alpha: float = 0.1
    learner_learning_rate: float = 1e-2
    learner_input_size: int = 28

@dataclass
class GeneratorConfig:
    generator_leaky_relu_alpha: float = 0.1
    generator_noise_size: int = 64
    generator_classes: int = 10
    generator_fc1_filters: int = 1024
    generator_fc2_filters: int = 128

    generator_conv1_filters: int = 64
    generator_conv2_filters: int = 1
    generator_output_size: int = 28