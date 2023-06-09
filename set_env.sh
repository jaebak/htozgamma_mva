export RUN_KERNEL=$(uname -r | cut -d '-' -f1)
export SET_ENV_PATH=set_env.sh
if [ "$RUN_KERNEL" == "18.5.0" ]; then
  # MAC OS
  source ~/Linux/root-6.26.10/bin/thisroot.sh
  source ../py-env/bin/activate
elif [ "$RUN_KERNEL" == "3.10.0" ]; then
  # CENTOS 7
  source ~/envSL7_gpu
  source py-env/bin/activate
fi
