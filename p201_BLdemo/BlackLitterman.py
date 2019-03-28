from matplotlib import pyplot
plt = pyplot
from pylab import *
import scipy.optimize
import pandas as pd
import numpy as np
import regex as re
from scipy.interpolate import spline

pd.set_option('expand_frame_repr', True)
pd.set_option('display.max_columns', 10)


# rf    无风险利率
# lmb   风险规避系数
# C     资产协方差矩阵
# V     资产差异（协方差矩阵中的对角线）
# W     资产权重
# R     资产收益
# mean  投资组合历史收益
# var   投资组合历史方差
# Pi    投资组合均衡超额收益
# tau   black-litterman的缩放因子

# XOM   埃克森美孚   AAPL    苹果      MSFT   微软
# JNJ   强生         GE    通用电气   GOOG    谷歌
# CVX   雪弗龙       PG     宝洁       WFC   富国银行


# 加载九家主要标准普尔公司的历史股票价格并返回它们截至2013-07-01的市值
def load_data():
    # symbols = ['XOM', 'AAPL', 'MSFT', 'JNJ', 'GE', 'GOOG', 'CVX', 'PG', 'WFC']
    # cap = {'XOM': 403.02e9, 'AAPL': 392.90e9, 'MSFT': 283.60e9, 'JNJ': 243.17e9, 'GE': 236.79e9,
    #        'GOOG': 292.72e9, 'CVX': 231.03e9, 'PG': 214.99e9, 'WFC': 218.79e9}
    # n = len(symbols)
    symbols = [
               'BSX',
               'FGJ',
               'PHZ',
                'PF3',
                'PF6',
               'PFB',
               'DCJ',
               'DFH',
               # 'AAPL',
               'CVX',
               'GOOG',
               #  'MSFT'
    ]
    cap = {
           'BSX': 5.90e9,
           'FGJ': 1.663e9,
           'PHZ': 7.339e9,
           'PF3': 25.10e9,
           'PF6': 25.982e9,
           'PFB': 25.69e9,
           'DCJ': 2.127e9,
           'DFH': 3.077e9,
           # 'AAPL':392.9e9,
           'CVX':2.3103e9,
           'GOOG': 2.9272e9,
           #  'MSFT': 283.6e9
    }
    prices_out, caps_out, date_out = [], [], []
    for s in symbols:
        dataframe = pd.read_csv('databank/%s.csv' % s, index_col=None, parse_dates=['date'])
        prices = list(dataframe['price'])[-240:] # 追溯240条数据
        date = list(dataframe['date'][-240:])
        # print(prices)
        date_out.append(date)
        prices_out.append(prices)
        caps_out.append(cap[s])
    return symbols, prices_out, caps_out, date_out


# 将历史股票价格与市值一起计算，并计算权重，历史收益和历史协方差
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
    return returns,expreturns, covars


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
        c_ = ({ 'type': 'eq',
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
def solve_weights(R, C, rf, stri):
    #
    # strs = "BSX<0.1"
    def fitness(W, R, C, rf):
        mean, var = port_mean_var(W, R, C)  # 计算投资组合的均值/方差
        util = (mean - rf) / sqrt(var)  # utility = 夏普比率
        return 1 / util  # 最大化utility，最小化其反向值

    def str_to_constraint(string):
        constraint=[{'type': 'eq', 'fun': lambda W1: sum(W1) - 1.}]
        symbols = {
            'BSX':0,
            'FGJ':1,
            'PHZ':2,
            'PF3':3,
            'PF6':4,
            'PFB':5,
            'DCJ':6,
            'DFH':7,
            # 'AAPL',
            'CVX':8,
            'GOOG':9
            #  'MSFT'
        }
        def append_obj(string):
            obj = re.match(r'(.*)>(.*)', string, re.M | re.I)
            if obj:
                compare='>'
            else:
                obj = re.match(r'(.*)<(.*)', string, re.M | re.I)
                compare='<'
            if compare=='>':
                objs = {'type': 'ineq', 'fun': lambda W1: W1[symbols[obj.group(1)]] - float(obj.group(2))}
            else:
                objs = {'type': 'ineq', 'fun': lambda W1: float(obj.group(2)) - W1[symbols[obj.group(1)]]}
            constraint.append(objs)

        while string !="":
            match = re.match(r'(.*?);(.*)',string, re.M | re.I)
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
    c_ = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})  # 权重总和必须为100％
    optimized = scipy.optimize.minimize(fitness, W, (R, C, rf), method='SLSQP', constraints=c_, bounds=b_)
    for i in range(n):
        optimized.x[i]=round(optimized.x[i],4)
    # print(optimized.x)
    W1=optimized.x
    # print(W1[0])
    b_1 = [(0., 1.) for i in range(n)]  # 边界的权重在0％... 100％之间。
    # c_1 = ( {'type': 'eq', 'fun': lambda W1: sum(W1) - 1.},
    #         {'type': 'ineq', 'fun': lambda W1: 0.1 - W1[0]},
    #         {'type': 'ineq', 'fun': lambda W1: 0.2 - W1[3]},  )
    c_1 = str_to_constraint(stri)
    optimized = scipy.optimize.minimize(fitness, W1, (R, C, rf), method='SLSQP', constraints=c_1, bounds=b_1)

    for i in range(n):
        optimized.x[i]=round(optimized.x[i],4)
    # print(optimized.x[0])
    # print(optimized.x)
    if not optimized.success: raise BaseException(optimized.message)
    return optimized.x


class Result:
    def __init__(self, W, tan_mean, tan_var, front_mean, front_var):
        self.W = W
        self.tan_mean = tan_mean
        self.tan_var = tan_var
        self.front_mean = front_mean
        self.front_var = front_var


def optimize_frontier(R, C, rf, st):
    W = solve_weights(R, C, rf, st)
    tan_mean, tan_var = port_mean_var(W, R, C)  # 计算相切投资组合
    front_mean, front_var = solve_frontier(R, C)  # 计算有效边界
    # 权重，相切组合资产均值和方差，有效前沿均值和方差
    return Result(W, tan_mean, tan_var, front_mean, front_var)


def create_views_and_link_matrix(names, views):
    r, c = len(views), len(names)
    if not r:
        P = zeros([1,c])
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

        # P = zeros([r, c])
        return array(Q), P

def str_to_views(string):
    views=[]
    def append_obj(string):
        obj = re.match(r'(.*)>(.*)=(.*)', string, re.M | re.I)
        if obj:
            compare='>'
        else:
            obj = re.match(r'(.*)<(.*)=(.*)', string, re.M | re.I)
            compare='<'
        objs = [obj.group(1),compare,obj.group(2),float(obj.group(3))]
        views.append(objs)

    while string !="":
        match = re.match(r'(.*?);(.*)',string, re.M | re.I)
        if match:
            append_obj(match.group(1))
            string = match.group(2)
        else:
            append_obj(string)
            string = ""
    return views



def solve_demo(returns,weight):
    rows, cols = returns.shape

    net = empty([rows, cols])
    ret = zeros(cols)
    for i in range(rows):
        net[i, 0] =  weight[i]
    for r in range(rows):
        for c in range(1,cols):
            returns[r, c] += 1
            net[r, c] = returns[r, c] * net[r, c-1]
    for c in range(cols):
        for r in range(rows):
            ret[c] += net[r, c]
    # print(ret[0])
    plt.subplot(212)
    t = np.arange(1,240,1)
    xnew = np.linspace(t.min(), t.max(), 1000)  # 300 represents number of points to make between T.min and T.max
    power_smooth = spline(t, ret, xnew)
    plt.plot(xnew, power_smooth)




def BL_model(string1,string2):
    names, prices, caps, dates= load_data()
    n = len(names)
    plt.figure(12,figsize=(12,9))
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
    # print(R)
    rf = .003  # 无风险利率
    # 计算投资组合历史收益和方差
    mean, var = port_mean_var(W, R, C)
    lmb = (mean - rf) / var  # 计算风险规避
    np.set_printoptions(suppress=True)
    Pi = dot(dot(lmb, C), W)  # 计算均衡超额收益
    views = str_to_views(string1)
    # res1 = optimize_frontier(R, C, rf)
    # res2 = optimize_frontier(Pi + rf, C, rf)
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
    res3 = optimize_frontier(Pi_adj + rf, C, rf, string2)



    # zhk1 = pd.DataFrame({'M-V权重': res1.W}, index=names).T
    # zhk2 = pd.DataFrame({'客观权重': res2.W}, index=names).T
    zhk3 = pd.DataFrame({'主观权重': res3.W}, index=names).T

    solve_demo(ret,res3.W)
    print(zhk3)

    # print(Pi+rf)
    # print(Pi_adj + rf)
    plt.subplot(222)
    # display_assets(names, R, C, color='blue')
    display_assets(names, Pi_adj + rf, C, color='blue')
    display_frontier(res3, label='Black-Litterman', color='royalblue')
    xlabel('variance $\sigma$'), ylabel('mean $\mu$'), legend()
    plt.subplot(221)
    labels = [
              'BSXDL',
              'FGJZYL',
              'PHZZGFJJ',
              'PBZP13F3',
              'PBZP13F6',
              'PBZP13PB',
              'DCJAZQ',
              'DFHCY',
              # 'AAPL',
              'CVXSZJ',
              'GOOGFJJ',
              # 'MSFT'
    ]
    plt.pie(res3.W,startangle = 90,shadow=False,autopct='%2.2f%%',pctdistance = 1.2)
    plt.axis('equal')
    plt.legend(loc='center left',bbox_to_anchor=(-0.2, 0.5),labels=labels)
    plt.show()
    return "资产配置完成！"


