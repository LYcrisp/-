import argparse

def ParseArgs():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
	parser.add_argument('--batch', default=1024, type=int, help='batch size')
	parser.add_argument('--kg_batch', default=4096, type=int, help='batch size for kg')
	parser.add_argument('--tstBat', default=1024, type=int, help='number of users in a testing batch')
	parser.add_argument('--reg', default=1e-7, type=float, help='weight decay regularizer')
	parser.add_argument('--epoch', default=25, type=int, help='number of epochs')
	parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
	parser.add_argument('--latdim', default=64, type=int, help='embedding size')
	parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
	parser.add_argument('--load_model', default=None, help='model name to load')
	parser.add_argument('--topk', default=20, type=int, help='K of top K')
	parser.add_argument('--data', default='yelp2018', type=str, help='name of dataset')
	parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
	parser.add_argument('--gpu', default='2', type=str, help='indicates which gpu to use')
	parser.add_argument('--layer_num_kg', default=1, type=int)
	parser.add_argument('--mess_dropout_rate', default=0.1, type=float)
	parser.add_argument('--ssl_reg', default=1e-1, type=float, help='weight for contrative learning')
	parser.add_argument('--temp', default=0.1, type=float, help='temperature in contrastive learning')
	parser.add_argument("--seed", type=int, default=421, help="random seed")

	parser.add_argument('--dims', type=str, default='[1000]')
	parser.add_argument('--d_emb_size', type=int, default=10)
	parser.add_argument('--norm', type=bool, default=True)
	parser.add_argument('--steps', type=int, default=5)
	parser.add_argument('--noise_scale', type=float, default=0.1)
	parser.add_argument('--noise_min', type=float, default=0.0001)
	parser.add_argument('--noise_max', type=float, default=0.02)
	parser.add_argument('--sampling_steps', type=int, default=0)

	parser.add_argument('--rebuild_k', type=int, default=1)
	parser.add_argument('--e_loss', type=float, default=0.5)

	parser.add_argument('--keepRate', type=float, default=0.5)
	parser.add_argument('--res_lambda', type=float, default=0.5)
	parser.add_argument('--triplet_num', type=int, default=10)
	parser.add_argument('--cl_pattern', type=int, default=0)


	#-------------------------new-------------------------
	parser.add_argument('--pop_alpha', type=float, default=0.3, help='popularity regularization weight')


	return parser.parse_args()
args = ParseArgs()