from matplotlib import pyplot
plt = pyplot
from pylab import *
import scipy.optimize
import pandas as pd
import numpy as np
import regex as re
from scipy import interpolate
# from cvxopt import solvers
import cvxopt

pd.set_option('expand_frame_repr', True)
pd.set_option('display.max_columns', 10)


# 加载数据
def load_data():
    symbols = ['BSX',
               'FGJ',
               'PHZ',
               'PF3',
               'PF6',
               'PFB',
               'DCJ',
               'DFH',
               'CVX',
               'GOOG',
               ]
    cap = {'BSX': 5.90e9,
           'FGJ': 1.663e9,
           'PHZ': 7.339e9,
           'PF3': 25.10e9,
           'PF6': 25.982e9,
           'PFB': 25.69e9,
           'DCJ': 2.127e9,
           'DFH': 3.077e9,
           'CVX': 2.3103e9,
           'GOOG': 2.9272e9,
           }
    prices_out, caps_out, date_out = [], [], []
    for s in symbols:
        dataframe = pd.read_csv('databank/%s.csv' % s, index_col=None, parse_dates=['date'])
        prices = list(dataframe['price'])[-240:]  # 追溯240条数据
        date = list(dataframe['date'][-240:])
        date_out.append(date)
        prices_out.append(prices)
        caps_out.append(cap[s])
    dataframe = pd.read_csv('point/userpoint1.csv', index_col=None)
    userid = list(dataframe['id'])
    userpoint = list(dataframe['point'])
    usercons = list(dataframe['constraint'])
    userlmbs = list(dataframe['lmb'])
    return symbols, prices_out, caps_out, date_out, userid, userpoint, usercons, userlmbs


# 将历史价格与市值一起计算，并计算权重，历史收益和历史协方差
def assets_historical_returns_and_covariances(prices):
    prices = matrix(prices)  # 为价格创建numpy矩阵
    # 创建历史收益矩阵
    rows, cols = prices.shape
    returns = empty([rows, cols - 1])
    for r in range(rows):
        for c in range(cols - 1):
            p0, p1 = prices[r, c], prices[r, c + 1]
            returns[r, c] = (p1 / p0) - 1
    # 计算收益
    expreturns = array([])
    for r in range(rows):
        expreturns = append(expreturns, np.mean(returns[r]))
    # 计算协方差
    covars = cov(returns)
    expreturns = (1 + expreturns) ** 360 - 1  # 年度回报
    covars = covars * 360  # 年度协方差
    return returns, expreturns, covars


# 计算投资组合平均收益
def port_mean(W, R):
    return sum(R * W)


# 计算收益的投资组合方差
def port_var(W, C):
    return dot(dot(W, C), W)


# port_mean和port_var的组合 - 收益计算的均值和方差
def port_mean_var(W, R, C):
    return port_mean(W, R), port_var(W, C)


# 给定无风险利率，资产收益率和协方差，此函数计算均值-方差边界并在两个数组中返回其[x，y]点
def solve_frontier(R, C):
    def fitness(W, R, C, r):
        # 对于给定的收益水平r，找到最小化投资组合方差的权重。
        mean, var = port_mean_var(W, R, C)
        penalty = 100 * abs(mean - r)  # 不符合规定的投资组合收益的惩罚作为优化的约束方式
        return var + penalty

    frontier_mean, frontier_var = [], []
    n = len(R)  # 投资组合中的资产数量
    for r in linspace(min(R), max(R), num=30):  # 迭代Y轴上的返回范围
        W = ones([n]) / n  # 以相同的权重开始优化
        b_ = [(0, 1) for i in range(n)]
        c_ = ({'type': 'eq',
               'fun': lambda W: sum(W) - 1.
               })
        optimized = scipy.optimize.minimize(fitness, W, (R, C, r), method='SLSQP', constraints=c_, bounds=b_)
        if not optimized.success:
            raise BaseException(optimized.message)
        # 为有效边界添加点[x,y] = [optimized.x, r]
        frontier_mean.append(r)
        frontier_var.append(port_var(optimized.x, C))
    return frontier_mean, frontier_var


# 给定无风险利率，资产收益率和协方差，此函数计算夏普比率最大化的相切投资组合的权重
def solve_weights(R, C, rf, stri,lmb):
    def fitness(W, R, C, rf):
        mean, var = port_mean_var(W, R, C)  # 计算投资组合的均值/方差
        util = (mean - rf) / sqrt(var)  # utility = 夏普比率
        return 1 / util  # 最大化utility，最小化其反向值

    def str_to_constraint(string):
        constraint = [{'type': 'eq', 'fun': lambda W1: sum(W1) - 1.}]
        symbols = {
            'BSX': 0,
            'FGJ': 1,
            'PHZ': 2,
            'PF3': 3,
            'PF6': 4,
            'PFB': 5,
            'DCJ': 6,
            'DFH': 7,
            'CVX': 8,
            'GOOG': 9
        }

        def append_obj(string):
            obj = re.match(r'(.*)>(.*)', string, re.M | re.I)
            if obj:
                compare = '>'
            else:
                obj = re.match(r'(.*)<(.*)', string, re.M | re.I)
                compare = '<'
            if compare == '>':
                objs = {'type': 'ineq', 'fun': lambda W1: W1[symbols[obj.group(1)]] - float(obj.group(2))}
            else:
                objs = {'type': 'ineq', 'fun': lambda W1: float(obj.group(2)) - W1[symbols[obj.group(1)]]}
            constraint.append(objs)

        while string != "":
            match = re.match(r'(.*?);(.*)', string, re.M | re.I)
            if match:
                append_obj(match.group(1))
                string = match.group(2)
            else:
                append_obj(string)
                string = ""
        return constraint

    n = len(R)
    W = ones([n]) / n  # 以相同的权重开始优化
    b_ = [(0., 1.) for i in range(n)]  # 边界的权重在0％... 100％之间。
    c_ = str_to_constraint(stri)
    constraint = [{'type': 'eq', 'fun': lambda W: sum(W) - 1.}]
    # optimized = scipy.optimize.minimize(fitness, W, (R, C, rf), method='SLSQP', constraints=constraint, bounds=b_)

    P = cvxopt.matrix(lmb * C)
    # print(P)
    G = cvxopt.matrix([[-1.,0,0,0,0,0,0,0,0,0],
                      [0,-1.,0,0,0,0,0,0,0,0],
                      [0,0,-1.,0,0,0,0,0,0,0],
                      [0,0,0,-1.,0,0,0,0,0,0],
                      [0,0,0,0,-1.,0,0,0,0,0],
                      [0,0,0,0,0,-1.,0,0,0,0],
                      [0,0,0,0,0,0,-1.,0,0,0],
                      [0,0,0,0,0,0,0,-1.,0,0],
                      [0,0,0,0,0,0,0,0,-1.,0],
                      [0,0,0,0,0,0,0,0,0,-1.]])
    # print(G)
    h = cvxopt.matrix([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    # print(h)
    A = cvxopt.matrix([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],(1,10))
    # print(A)
    b = cvxopt.matrix([1.0])
    # print(R)
    d = transpose(R)
    Q = cvxopt.matrix([ 0.14729798, 0.22814802, 0.44950883, 0.00695611, 0.00337905, 0.00329889,
 -0.0009483,   0.32112076,  0.02399866,  0.09008425])
    # print(Q)

    result = cvxopt.solvers.qp(P, Q, G, h, A, b)
    print(result['x'])
    return
    # for i in range(n):
    #     optimized.x[i] = round(optimized.x[i], 4)
    # if not optimized.success: raise BaseException(optimized.message)
    # return optimized.x


class Result:
    def __init__(self, W, front_mean, front_var):
        self.W = W
        # self.tan_mean = tan_mean
        # self.tan_var = tan_var
        self.front_mean = front_mean
        self.front_var = front_var


def optimize_frontier(R, C, rf, st,lmb):
    W = solve_weights(R, C, rf, st,lmb)
    # tan_mean, tan_var = port_mean_var(W, R, C)  # 计算相切投资组合
    front_mean, front_var = solve_frontier(R, C)  # 计算有效边界
    # 权重，相切组合资产均值和方差，有效前沿均值和方差
    return Result(W, front_mean, front_var)


def create_views_and_link_matrix(names, views):
    r, c = len(views), len(names)
    if not r:
        P = zeros([1, c])
        Q = [0.]
        return Q, P
    else:
        Q = [views[i][3] for i in range(r)]  # view matrix
        P = zeros([r, c])
        nameToIndex = dict()

        for i, n in enumerate(names):
            nameToIndex[n] = i

        for i, v in enumerate(views):
            name1, name2 = views[i][0], views[i][2]
            P[i, nameToIndex[name1]] = +1 if views[i][1] == '>' else -1
            P[i, nameToIndex[name2]] = -1 if views[i][1] == '>' else +1

        return array(Q), P


def str_to_views(string):
    views = []

    def append_obj(string):
        obj = re.match(r'(.*)>(.*)=(.*)', string, re.M | re.I)
        if obj:
            compare = '>'
        else:
            obj = re.match(r'(.*)<(.*)=(.*)', string, re.M | re.I)
            compare = '<'
        objs = [obj.group(1), compare, obj.group(2), float(obj.group(3))]
        views.append(objs)

    while string != "":
        match = re.match(r'(.*?);(.*)', string, re.M | re.I)
        if match:
            append_obj(match.group(1))
            string = match.group(2)
        else:
            append_obj(string)
            string = ""
    return views


def solve_demo(returns, weight):
    rows, cols = returns.shape
    net = empty([rows, cols])
    ret = zeros(cols)
    for i in range(rows):
        net[i, 0] = weight[i]
    for r in range(rows):
        for c in range(1, cols):
            returns[r, c] += 1
            net[r, c] = returns[r, c] * net[r, c - 1]
    for c in range(cols):
        for r in range(rows):
            ret[c] += net[r, c]
    plt.subplot(212)
    t = np.arange(1, 240, 1)
    xnew = np.linspace(t.min(), t.max(), 1000)
    tck = interpolate.splrep(t, ret)
    ynew = interpolate.splev(xnew, tck)
    plt.plot(xnew, ynew)


# 主函数
def BL_model():
    labels = [
        'BSXDL',
        'FGJZYL',
        'PHZZGFJJ',
        'PBZP13F3',
        'PBZP13F6',
        'PBZP13PB',
        'DCJAZQ',
        'DFHCY',
        'CVXSZJ',
        'GOOGFJJ',
    ]

    def BL(names, prices, caps, userid, userpoint, usercons, lmb):
        n = len(names)
        plt.figure(12, figsize=(12, 9))

        def display_assets(names, R, C, color='black'):
            scatter([C[i, i] ** .5 for i in range(n)], R, marker='x', color=color), grid(True)  # 画资产
            for i in range(n):
                text(C[i, i] ** .5, R[i], '  %s' % names[i], verticalalignment='center', color=color)  # 画标签

        def display_frontier(result, label=None, color='black'):
            plt.text(result.tan_var ** .5, result.tan_mean, '   tangent', verticalalignment='center', color='red')
            plt.scatter(result.tan_var ** .5, result.tan_mean, marker='o', color='red'), grid(True)
            plt.plot(np.power(result.front_var, 0.5), result.front_mean, label=label, color=color), grid(True)  # 画有效边界

        W = array(caps) / sum(caps)  # 计算股票市场权重
        ret, R, C = assets_historical_returns_and_covariances(prices)
        rf = .003  # 无风险利率
        # 计算投资组合历史收益和方差
        # mean, var = port_mean_var(W, R, C)


        # lmb = (mean - rf) / var  # 计算风险规避



        np.set_printoptions(suppress=True)
        Pi = dot(dot(lmb, C), W)  # 计算均衡超额收益
        views = str_to_views(userpoint)
        tau = .025  # 缩放因子
        sub_a = inv(dot(tau, C))
        sub_c = dot(inv(dot(tau, C)), Pi)
        if views:
            Q, P = create_views_and_link_matrix(names, views)
            # 计算主观因素的不确定性矩阵omega
            omega = dot(dot(dot(tau, P), C), transpose(P))  # [(0.025 * P) * C] * P转置
            sub_b = dot(dot(transpose(P), inv(omega)), P)
            sub_d = dot(dot(transpose(P), inv(omega)), Q)
            # 合并主客观因素，计算均衡超额收益
            Pi_adj = dot(inv(sub_a + sub_b), (sub_c + sub_d))
        else:
            Pi_adj = dot(inv(sub_a), sub_c)
        # 计算用户资产配置权重
        res = optimize_frontier(Pi_adj + rf, C, rf, usercons,lmb)
        # userW = pd.DataFrame({userid: res.W}, index=names).T
        # # 绘制资产配置图
        # solve_demo(ret, res.W)
        # plt.subplot(222)
        # display_assets(names, Pi_adj + rf, C, color='blue')
        # display_frontier(res, label='Black-Litterman', color='royalblue')
        # xlabel('variance $\sigma$'), ylabel('mean $\mu$'), legend()
        # plt.subplot(221)
        # plt.pie(res.W, startangle=90, shadow=False, autopct='%2.2f%%', pctdistance=1.2)
        # plt.axis('equal')
        # plt.legend(loc='center left', bbox_to_anchor=(-0.2, 0.5), labels=labels)
        # plt.savefig('BLimage/%s.png' % userid)
        # plt.show()
        # print(port_var(res.W,C))
        # return userW
        return 1
    names, prices, caps, dates, userid, userpoints, usercons, userlmbs = load_data()
    num = len(userid)
    userWeight = BL(names, prices, caps, userid[0], userpoints[0], usercons[0], userlmbs[0])
    for i in range(1, num):
        # print(userid[i], userpoints[i], usercons[i], userlmbs[i])
        res = BL(names, prices, caps, userid[i], userpoints[i], usercons[i], userlmbs[i])
        # userWeight = pd.concat([userWeight, res])
    # userWeight.to_csv('point/userPortfolio.csv', index=True, sep=',')
    return


BL_model()