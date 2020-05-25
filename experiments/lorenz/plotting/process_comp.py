import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

import pandas as pd
import os
opj = os.path.join

def main(pemdir, ceemdir):

	N = 10

	ceem_sigmas = np.zeros((100,N))
	ceem_rhos = np.zeros((100,N))
	ceem_betas = np.zeros((100,N))

	ceem_tpe = np.zeros((N,))

	for i, seed in enumerate(range(43,43+N)):
		file = opj(ceemdir, 'seed%d'%seed, 'progress.csv')
		df = pd.read_csv(file)
		sigmas = df['test/sigma'].values
		if sigmas.shape[0] < 100:
			T_ = 100 - sigmas.shape[0]
			sigmas = np.concatenate([sigmas, np.ones((T_,)) * sigmas[-1]])
		betas = df['test/beta'].values
		if betas.shape[0] < 100:
			T_ = 100 - betas.shape[0]
			betas = np.concatenate([betas, np.ones((T_,)) * betas[-1]])
		rhos = df['test/rho'].values
		if rhos.shape[0] < 100:
			T_ = 100 - rhos.shape[0]
			rhos = np.concatenate([rhos, np.ones((T_,)) * rhos[-1]])
		ceem_sigmas[:,i] = sigmas
		ceem_betas[:,i] = betas
		ceem_rhos[:,i] = rhos


		# max smooth time per epoch, since parallelized
		ceem_tpe[i] += df.filter(items=['smooth/time/%d'%j for j in range(4)]).max(axis=1).sum()
	
		ceem_tpe[i] += df['learn/time'].sum()
		ceem_tpe[i] /= len(df)


	pem_sigmas = np.zeros((100,N))
	pem_rhos = np.zeros((100,N))
	pem_betas = np.zeros((100,N))
	pem_tpe = np.zeros((N,))

	for i, seed in enumerate(range(43,43+N)):
		file = opj(pemdir, 'seed%d'%seed, 'progress.csv')
		df = pd.read_csv(file)
		pem_sigmas[:,i] = df['test/sigma'].values[:-1]
		pem_betas[:,i] = df['test/beta'].values[:-1]
		pem_rhos[:,i] = df['test/rho'].values[:-1]

		pem_tpe[i] = df['train/Etime'].sum() + df['train/Mtime'].sum()
		pem_tpe[i] /= len(df)

	print('PEM tpe/CEEM tpe = %.3f' % (pem_tpe.mean() / ceem_tpe.mean()))

	plt.figure(figsize=(4,4))

	x = np.arange(100)
	ax1 = plt.subplot(3,2,1)
	plt.plot(x, ceem_sigmas , alpha=0.5)
	plt.plot([0,99],[10.,10.], 'k--')
	plt.ylabel(r'$\sigma$')
	plt.title('CE-EM')


	x = np.arange(100)
	ax2 = plt.subplot(3,2,3, sharex=ax1)
	plt.plot(x, ceem_rhos , alpha=0.5,)
	plt.plot([0,99],[28., 28.], 'k--')
	plt.ylabel(r'$\rho$')

	x = np.arange(100)		
	ax3 = plt.subplot(3,2,5, sharex=ax1)
	plt.plot(x, ceem_betas , alpha=0.5,)
	plt.plot([0,99],[8./3., 8./3.], 'k--')
	plt.ylabel(r'$\beta$')
	plt.xlabel('EM Epoch')

	x = np.arange(100)
	ax4 = plt.subplot(3,2,2, sharey=ax1)
	plt.plot(x, pem_sigmas , alpha=0.5,)
	plt.plot([0,99],[10.,10.], 'k--')
	plt.title('Particle EM')
	
	ax4.set_ylim((np.array([0.9, 1.1])*10.).tolist())

	x = np.arange(100)
	ax5 = plt.subplot(3,2,4, sharey=ax2, sharex=ax4)
	plt.plot(x, pem_rhos , alpha=0.5)
	plt.plot([0,99],[28., 28.], 'k--')
	

	ax5.set_ylim((np.array([0.9, 1.1])*28.).tolist())

	x = np.arange(100)		
	ax6 = plt.subplot(3,2,6, sharey=ax3, sharex=ax4)
	plt.plot(x, pem_betas , alpha=0.5)
	plt.plot([0,99],[8./3., 8./3.], 'k--')
	plt.xlabel('EM Epoch')

	ax6.set_ylim((np.array([0.9, 1.1])*8./3.).tolist())
	ax1.set_xlim([0,10])
	ax4.set_xlim([-2,100])

	plt.tight_layout()

	plt.savefig('LorenzComp.pdf')

	plt.show()



if __name__ == '__main__':
	pemdir = './data/lorenz/comp/pem'
	ceemdir = './data/lorenz/comp/ceem'
	main(pemdir, ceemdir)