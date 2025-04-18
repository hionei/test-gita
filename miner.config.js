module.exports = {
  apps: [
    {
      name: 'miner',
      interpreter: 'python3',
      script: './neurons/miner.py',
      args: '--netuid 50 --logging.debug --logging.trace --wallet.name synth1-coldkey --wallet.hotkey synth1-hotkey --axon.port 9999 --blacklist.force_validator_permit true --blacklist.validator_min_stake 0',
      env: {
        PYTHONPATH: '.'
      },
    },
  ],
};
