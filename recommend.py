import mf
import sys

argc = len(sys.argv)
if argc < 5:
    print "usage: python train.py $traing_examples $recommend_count $num_users, $num_items [, $num_factors, $num_iterations, $reg_param, $conv_loss_value"
    sys.exit()

fexamples = sys.argv[1]
reco_cnt = int(sys.argv[2])
num_users = int(sys.argv[3])
num_items = int(sys.argv[4])
num_factors = argc >= 6 and int(sys.argv[5]) or 40
num_iterations = argc >= 7 and int(sys.argv[6]) or 30
reg_param = argc >= 8 and float(sys.argv[7]) or 0.8
conv_loss_value = argc >= 9 and float(sys.argv[8]) or 0

counts = mf.load_matrix(fexamples, num_users, num_items)
imcf = mf.ImplicitMF(counts, num_factors, num_iterations, conv_loss_value, reg_param)
imcf.train_model()
for user in range(0, num_users):
    reco_items = imcf.recommend(user, reco_cnt)
    for item in reco_items:
        print("%d\t%d" % (user, item))
