import matplotlib.pyplot as plt
"""
绘图代码，绘制指标图
"""


# 定义可选的线条颜色列表（最多8种）
colors = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']

def SubGraph(s1, s2, loc):
	"""
	根据行数 s1、列数 s2 和图位置 loc，生成 subplot 的编号格式
	即将 (s1, s2, loc) 拼接为三位数，例如 (2,1,1) -> 211 表示两行一列中的第一个子图
	"""
	return int(str(s1) + str(s2) + str(loc))

def PlotOneLine(x, y, color, subId=None):
	"""
	绘制单条曲线。
	参数：
	- x, y: 曲线的数据点
	- color: 线条颜色（字符，如 'r' 表示红色）
	- subId: 子图编号，如果指定则在对应 subplot 中绘图
	"""
	if subId != None:
		ax = plt.subplot(subId)  # 在指定的 subplot 中绘图
	if color:
		plt.plot(x, y, color)  # 绘制有颜色指定的曲线
	else:
		plt.plot(x, y)  # 绘制默认颜色曲线
	plt.show()  # 显示图像（注意：这里 show 是在每条线后调用）

def PlotLineChart(x, y, xName = '', yName = '', subGraph=True):
	"""
	绘制折线图函数，支持单图或多子图绘制。
	参数：
	- x, y: 输入数据（可为一维或二维数组），一维表示单条线，二维表示多条线
	- xName, yName: 坐标轴名称（未被使用）
	- subGraph: 是否使用子图方式分别绘制每条曲线

	注意事项：
	- 若 x 和 y 形状不一致，或维度大于2，则报错
	- 若不使用子图且曲线条数超过颜色种类，将提示错误
	"""
	if x.shape != y.shape or len(x.shape) > 2:
		# 输入数据形状不匹配或维度过高，打印错误信息
		print(x.shape, y.shape)
		print('Input Data Error for Plotter')
		return

	plt.figure(1)  # 创建图像窗口1

	if subGraph:
		# 使用子图方式逐条绘制
		if len(x.shape) == 2:
			# 多条曲线绘制，每条单独子图
			for i in range(x.shape[0]):
				subId = SubGraph(x.shape[0], 1, i + 1)  # 每行一个子图
				PlotOneLine(x[i], y[i], subId)
		else:
			# 单条曲线，单独一个子图
			subId = SubGraph(1, 1, 1)
			PlotOneLine(x, y, subId)
	else:
		# 所有曲线画在同一个图上，用不同颜色区分
		if len(x.shape) == 2 and x[0] > len(colors):
			# 曲线数超过可用颜色数，报错建议使用子图
			print('Too Many Curve, Use SubGraph')
			return
		for i in range(x.shape[0]):
			PlotOneLine(x[i], y[i], colors[i])  # 多条曲线，依次上色绘制

	plt.show()  # 最后统一显示图像
