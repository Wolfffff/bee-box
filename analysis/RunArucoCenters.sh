for VARIABLE in `ls`; do python ../ArucoCenters.py --type DICT_4X4_100 --image ${VARIABLE} | cut -d '_' -f 8 ; done > BaslerTest.csv
