from torchrl.record import CSVLogger

logger = CSVLogger(exp_name="my_exp", log_dir="workdir/csv_logs", video_format="mp4")

logger.log_scalar("my_scalar", 0.4)

from torchrl.envs import GymEnv

env = GymEnv("CartPole-v1", from_pixels=True, pixels_only=False)

print(env.rollout(max_steps=3))

from torchrl.envs import TransformedEnv

from torchrl.record import VideoRecorder

recorder = VideoRecorder(logger, tag="my_video")
record_env = TransformedEnv(env, recorder)

rollout = record_env.rollout(max_steps=3)

# Uncomment this line to save the video on disk:
recorder.dump()
