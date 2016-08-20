# packages
APT_PKGS := python-pip python-dev
BREW_PKGS := --python
PIP_PKGS := numpy scipy pandas scikit-learn

SED := sed

# directories
DIR_DATA := data
DIR_BUILD := build
DIR_BIN := $(DIR_BUILD)/bin
DIR_BLEND := $(DIR_BUILD)/blend
DIR_FEATURE := $(DIR_BUILD)/feature
DIR_METRIC := $(DIR_BUILD)/metric
DIR_MODEL := $(DIR_BUILD)/model

# directories for the cross validation and ensembling
DIR_VAL := $(DIR_BUILD)/val
DIR_TST := $(DIR_BUILD)/tst

DIRS := $(DIR_DATA) $(DIR_BUILD) $(DIR_FEATURE) $(DIR_METRIC) $(DIR_MODEL) \
        $(DIR_VAL) $(DIR_TST) $(DIR_BIN) $(DIR_BLEND)

# data files for training and predict
DATA_TRN := $(DIR_DATA)/gender_age_train.csv
DATA_TST := $(DIR_DATA)/gender_age_test.csv
DATA_APP_EVENT := $(DIR_DATA)/app_events.csv
DATA_APP_LABEL := $(DIR_DATA)/app_labels.csv
DATA_EVENT := $(DIR_DATA)/events.csv
DATA_LABEL := $(DIR_DATA)/label_categories.csv
DATA_PHONE := $(DIR_DATA)/phone_brand_device_model.csv
SAMPLE_SUBMISSION := $(DIR_DATA)/sample_submission.csv

ID_TST := $(DIR_DATA)/id.tst.csv
HEADER := $(DIR_DATA)/header.csv
CV_ID := $(DIR_DATA)/cv_id.txt

Y_TRN:= $(DIR_FEATURE)/y.trn.txt
Y_TST:= $(DIR_FEATURE)/y.tst.txt

$(DIRS):
	mkdir -p $@

$(HEADER): $(SAMPLE_SUBMISSION)
	head -1 $< > $@

$(ID_TST): $(SAMPLE_SUBMISSION)
	cut -d, -f1 $< | tail -n +2 > $@

$(Y_TST): $(SAMPLE_SUBMISSION) | $(DIR_FEATURE)
	cut -d, -f2 $< | tail -n +2 > $@

$(Y_TRN) $(CV_ID): $(DATA_TRN) | $(DIR_FEATURE)
	python src/extract_target.py --train-file $< \
                                 --target-file $(Y_TRN) \
                                 --cvid-file $(CV_ID)

# cleanup
clean::
	find . -name '*.pyc' -delete

clobber: clean
	-rm -rf $(DIR_DATA) $(DIR_BUILD)

.PHONY: clean clobber mac.setup ubuntu.setup apt.setup pip.setup
