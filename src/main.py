from re import T
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal, wishart, dirichlet  # 多次元ガウス分布、ウィシャー卜分布、ディリクレ分布
import matplotlib.pyplot as plt


# GMMのギブスサンプリング
# 参考：https://www.anarchive-beta.com/entry/2020/11/28/210948

class GMM_GibbsSampling():
    def __init__(self):
        self.definition_observation_model()
        self.generation_data()
        self.definition_conjugate_prior()
        self.definition_initial_value()
        self.gibbs_sampling()
        pass

    def definition_observation_model(self):
        # 観測モデルの設定 ########################################################
        ## 次元数の設定 (2次元のグラフで表現)
        self.D = 2

        ## クラスタ数
        self.K = 3

        ## K個の多次元ガウス分布の真の平均ベクトルを設定
        self.mu_truth_list = np.array(
            [[5.0, 35.0],
             [-20.0, -10.0],
             [30.0, -20.0]]
        )

        ## K個の多次元ガウス分布の真の分散共分散行列を設定
        self.sigma2_truth_list = np.array(
            [[[250.0, 65.0], [65.0, 270.0]],
             [[125.0, -45.0], [-45.0, 175.0]],
             [[210.0, -15.0], [-15.0, 250.0]]]
        )

        ## 真の混合比率の設定 (全て足すと1になる)
        self.pi_truth_list = np.array([0.45, 0.25, 0.3])
        # 観測モデルの確認 ########################################################
        # K
        # K個のクラスタそれぞれで平均値から標準偏差の3倍を引いた値と足した値を計算し、その最小値から最大値までを範囲
        ## 作図用のx軸のxの値を作成 (指定した要素数で等間隔に区切る)
        x_1_line = np.linspace(
            np.min(self.mu_truth_list[:, 0] - 3 * np.sqrt(self.sigma2_truth_list[:, 0, 0])),
            np.max(self.mu_truth_list[:, 0] + 3 * np.sqrt(self.sigma2_truth_list[:, 0, 0])),
            num=300
        )

        ## 作図用のy軸のxの値を作成 (指定した要素数で等間隔に区切る)
        x_2_line = np.linspace(
            np.min(self.mu_truth_list[:, 1] - 3 * np.sqrt(self.sigma2_truth_list[:, 1, 1])),
            np.max(self.mu_truth_list[:, 1] + 3 * np.sqrt(self.sigma2_truth_list[:, 1, 1])),
            num=300
        )

        ## 作図用の格子状の点を作成
        self.x_1_grid, self.x_2_grid = np.meshgrid(x_1_line, x_2_line)

        ## 作図用のxの点を作成
        x_point = np.stack([self.x_1_grid.flatten(), self.x_2_grid.flatten()], axis=1)

        ## 作図用に各次元の要素数を保存
        self.x_dim = self.x_1_grid.shape

        # 観測モデルの確率密度を計算 (確率密度は確率を面積、確率変数を幅としたときの高さに相当)########################################################
        self.true_model = 0

        for k in range(self.K):
            # クラスタkの分布の確率密度を計算 (x)
            tmp_density = multivariate_normal.pdf(
                x=x_point, mean=self.mu_truth_list[k], cov=self.sigma2_truth_list[k]
            )
            # K個の分布の加重平均を計算 (混合ガウス分布の潜在表現がなしの表現)
            self.array = np.array([0.45, 0.25, 0.3])
            self.true_model += self.pi_truth_list[k] * tmp_density

        ## 観測モデルの作図
        plt.figure(figsize=(12, 9))
        plt.contour(self.x_1_grid, self.x_2_grid, self.true_model.reshape(self.x_dim))
        plt.suptitle('Gaussian Mixture Model', fontsize=20)
        plt.title('K=' + str(self.K), loc='left')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.colorbar()
        # plt.show()
        return

    def generation_data(self):
        # サンプル数
        self.N = 250

        # 潜在変数の設定
        self.z_truth = np.random.multinomial(n=1, pvals=self.pi_truth_list, size=self.N)

        # クラスタ番号の抽出
        _, self.z_truth_n = np.where(self.z_truth == 1)

        # サンプルの生成 (K個の多次元ガウスからサンプリング、データによってクラスタ異なる)
        self.x_nd = np.array(
            [np.random.multivariate_normal(mean=self.mu_truth_list[k], cov=self.sigma2_truth_list[k], size=1
                                           ).flatten() for k in self.z_truth_n]
            )

        # 観測データの散布図を作成
        plt.figure(figsize=(12, 9))
        for k in range(self.K):
            k_idx = np.where(self.z_truth_n == k)
            plt.scatter(x=self.x_nd[k_idx, 0], y=self.x_nd[k_idx, 1], label='cluster:' + str(k + 1))
        plt.contour(self.x_1_grid, self.x_2_grid, self.true_model.reshape(self.x_dim), linestyle='--')
        plt.suptitle('Gaussian Mixture Model', fontsize=20)
        plt.title('$N=' + str(self.N) + ', K=' + str(self.K) + ', \pi=[' + ','.join(
            [str(pi) for pi in self.pi_truth_list]) + ']$', loc='left')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.colorbar()
        plt.show()
        return

    def definition_conjugate_prior(self):
        # muの事前分布 (多次元ガウス分布)のハイパーパラメータ
        self.beta = 1.0
        self.m_d = np.repeat(0.0, self.D)

        # muの事前分布

        # lambdaの事前分布 (ウィシャー卜分布)のハイパーパラメータ
        self.w_dd = np.identity(self.D) * 0.0005
        self.nu = self.D

        # piの事前分布 (ディリクレ分布)のハイパーパラメータ
        self.alpha_k = np.repeat(2.0, self.K)

    def definition_initial_value(self):
        # 観測モデルのパラメータをサンプル (入れ物を用意)
        self.mu_kd = np.empty((self.K, self.D))
        self.lambda_kdd = np.empty((self.K, self.D, self.D))

        for k in range(self.K):
            # クラスタKの精度行列をサンプル (クラスタごとに生成するためsizeは1)
            self.lambda_kdd[k] = wishart.rvs(df=self.nu, scale=self.w_dd, size=1)

            # クラスタKの平均ベクトルをサンプル
            self.mu_kd[k] = np.random.multivariate_normal(
                mean=self.m_d, cov=np.linalg.inv(self.beta * self.lambda_kdd[k])
            ).flatten()

        # 混合比率πのサンプル (どの多次元ガウス分布を選択するかを決めるパラメータ)
        self.pi_k = dirichlet.rvs(self.alpha_k, size=1).flatten()

    def gibbs_sampling(self):
        # パラメータの初期値を用いて，クラスタZの事後分布を計算
        # そこからZをサンプル

        # 試行回数の設定
        Iteration = 150

        # パラメータを初期化
        eta_nk = np.zeros((self.N, self.K))  # クラスタzの更新後のパラメータeta
        s_nk = np.zeros((self.N, self.K))
        beta_hat_k = np.zeros(self.K)
        m_hat_kd = np.zeros((self.K, self.D))
        w_hat_kdd = np.zeros((self.K, self.D, self.D))
        nu_hat_k = np.zeros(self.K)
        alpha_hat_k = np.zeros(self.K)

        # 推移確認用の受け皿 
        trace_s_in = [np.repeat(np.nan, self.N)]
        trace_mu_ikd = [self.mu_kd.copy()]
        trace_lambda_ikdd = [self.lambda_kdd.copy()]
        trace_pi_ik = [self.pi_k.copy()]
        trace_beta_ik = [np.repeat(self.beta, self.K)]
        trace_m_ikd = [np.repeat(self.m_d.reshape((1, self.D)), self.K, axis=0)]
        trace_w_ikdd = [np.repeat(self.w_dd.reshape((1, self.D, self.D)), self.K, axis=0)]
        trace_nu_ik = [np.repeat(self.nu, self.K)]
        trace_alpha_ik = [self.alpha_k.copy()]

        # ギブスサンプリング
        for i in range(Iteration):
            for k in range(self.K):
                # 潜在変数Zの事後分布のパラメータを計算．z_nの条件付き分布 (カテゴリ分布)のパラメータη_n
                # まずは，一項目
                tmp_eta_n = np.diag(
                    -0.5 * (self.x_nd - self.mu_kd[k]).dot(self.lambda_kdd[k]).dot((self.x_nd - self.mu_kd[k]).T)
                ).copy()  # Nコの全ての要素を一度で処理し，1からNの組み合わせで計算．その後，実際に必要なのは対角成分 (1, 1)の部分が必要なのでnp.diagで取り出す

                # 二項目 (対数の計算でln0とならないように微小値を入れている (いる？))
                tmp_eta_n += 0.5 * np.log(np.linalg.det(self.lambda_kdd[k]) + 1e-7)

                # 指数でくくる
                eta_nk[:, k] = np.exp(tmp_eta_n)
            # 正規化
            eta_nk /= np.sum(eta_nk, axis=1, keepdims=True)

            # 潜在変数z_nをサンプル
            for n in range(self.N):
                s_nk[n] = np.random.multinomial(n=1, pvals=eta_nk[n], size=1).flatten()

            # 観測モデルのパラメータをサンプル
            for k in range(self.K):
                # muの事後分布のパラメータを計算
                beta_hat_k[k] = np.sum(s_nk[:, k]) + self.beta  # ハイパーパラメータβの更新
                m_hat_kd[k] = np.sum(s_nk[:, k] * self.x_nd.T, axis=1)
                m_hat_kd += self.beta * self.m_d
                m_hat_kd /= beta_hat_k[k]  # ハイパーパラメータmの更新

                # lambdaの事後分布のパラメータを計算
                # 一項目
                tmp_w_dd = np.dot((s_nk[:, k] + self.x_nd.T), self.x_nd)
                # 二項目
                tmp_w_dd += self.beta * np.dot(self.m_d.reshape(self.D, 1), self.m_d.reshape(1, self.D))
                # 三項目
                tmp_w_dd -= beta_hat_k[k] * np.dot(m_hat_kd[k].reshape(self.D, 1), m_hat_kd[k].reshape(1, self.D))
                # 四項目
                tmp_w_dd += np.linalg.inv(self.w_dd)
                w_hat_kdd[k] = np.linalg.inv(tmp_w_dd)

                nu_hat_k[k] = np.sum(s_nk[:, k]) + self.nu

                # lambdaをサンプル (ウィシャート分布)
                self.lambda_kdd[k] = wishart.rvs(size=1, df=nu_hat_k[k], scale=w_hat_kdd[k])

                # muをサンプル (多次元ガウス分布)
                self.mu_kd[k] = np.random.multivariate_normal(
                    mean=m_hat_kd[k], cov=np.linalg.inv(beta_hat_k[k] * self.lambda_kdd[k]), size=1
                ).flatten()

            # 混合比率πのパラメータを計算
            alpha_hat_k = np.sum(s_nk, axis=0) + self.alpha_k  # αの更新

            # πをサンプル
            pi_k = dirichlet.rvs(size=1, alpha=alpha_hat_k).flatten()

            # 値を記録
            _, s_n = np.where(s_nk == 1)
            trace_s_in.append(s_n.copy())
            trace_mu_ikd.append(self.mu_kd.copy())
            trace_lambda_ikdd.append(self.lambda_kdd.copy())
            trace_pi_ik.append(pi_k.copy())
            trace_beta_ik.append(beta_hat_k.copy())
            trace_m_ikd.append(m_hat_kd.copy())
            trace_w_ikdd.append(w_hat_kdd.copy())
            trace_nu_ik.append(nu_hat_k.copy())
            trace_alpha_ik.append(alpha_hat_k.copy())

            # 動作確認
            # print(str(i + 1) + ' (' + str(np.round((i + 1) / Iteration * 100, 1)) + '%)')


if __name__ == '__main__':
    gmm_gibbs = GMM_GibbsSampling()