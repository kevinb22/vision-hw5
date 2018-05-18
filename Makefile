run_all: set_up run_23 test_batch_size test_learning_rate test_decreasing_learning_rate

run_21: FORCE
	python main.py --model LazyNet --epochs 50 --logdir run_21 --cuda True

# change based on whether or not we want to run with activation.
run_22: FORCE
	python main.py --model BoringNet --epochs 50 --logdir run_22_activation --cuda True

run_23: FORCE
	python main.py --model CoolNet --epochs 50 --logdir run_23 --cuda True

test_batch_size: FORCE
	python main.py --model CoolNet --epoch 50 --batchSize 16 --logdir batch_16 --cuda True

	python main.py --model CoolNet --epoch 50 --batchSize 32 --logdir batch_32 --cuda True

	python main.py --model CoolNet --epoch 50 --batchSize 64 --logdir batch_64 --cuda True

	python main.py --model CoolNet --epoch 50 --batchSize 128 --logdir batch_128 --cuda True

	python main.py --model CoolNet --epoch 50 --batchSize 256 --logdir batch_256 --cuda True

test_learning_rate: FORCE
	python main.py --lr 10 --model CoolNet --epoch 50  --logdir lr_10 --cuda True

	python main.py --lr 0.1 --model CoolNet --epoch 50 --logdir lr_0.1 --cuda True

	python main.py --lr 0.01 --model CoolNet --epoch 50 --logdir lr_0.01 --cuda True

	python main.py --lr 0.0001 --model CoolNet --epoch 50 --logdir lr_0.0001 --cuda True

test_decreasing_learning_rate: FORCE
	python main.py --model CoolNet --epoch 150

set_up: FORCE
	mkdir -p records/

# Phony target to force clean
FORCE: ;

