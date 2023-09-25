# Pretrained Emulators 

The subdirectory [``emulationResults/trainedParameters``](emulationResults/trainedParameters) stores pretrained emulation parameters. The commands below show how to use these parameters to make predictions on the test data.

## *TwistingCube*
Predictions can be run as follows:
```
python -m main --mode="evaluate" --data_path="TwistingBeam" --dir_label="_preTrained" --trained_params_dir="emulationResults/trainedParameters/TwistingBeam/"
```

## *Liver*
Predictions can be run as follows:
```
python -m main --mode="evaluate" --data_path="Liver" --dir_label="_preTrained" --trained_params_dir="emulationResults/trainedParameters/Liver/"
```

## *LeftVentricle*
Predictions can be run as follows:

```
python -m main --mode="evaluate" --data_path="LeftVentricle" --dir_label="_preTrained" --trained_params_dir="emulationResults/trainedParameters/LeftVentricle/"
```

Note, as discussed in the [``README``](README.md), this is a slightly different version of the *LeftVentricle* data that was used in the paper.
