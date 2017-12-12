import os; 

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cu
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import numpy as np

import model as m

from docopt import docopt

class ReprClassifier(nn.Module):
	'''
	This NN only used pretrained sentence representations (or more a subset of those dimensions).
	'''

	def __init__(self, input_dim, hidden_dim, output_dim, dropout):
		'''
		Initialize new classifier 

		:param input_dim 	Amount of dimensions that are used per sentence
		:param hidden_dim 	Amount of hidden units in a two layer MLP
		:param output_dim 	Amount of labels
		:param dropout 		Dropout applied on the output of the hidden layers
		'''
		super(ReprClassifier, self).__init__()
		self.hidden1 = nn.Linear(input_dim * 4, hidden_dim) # multiplication because of feature concatenation
		self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
		self.hidden3 = nn.Linear(hidden_dim, output_dim)
		self.dropout1 = nn.Dropout(p=dropout)
		self.dropout2 = nn.Dropout(p=dropout)

	def forward(self, repr_p, repr_h):
		
		# use feature concatenation
		feedforward_input = torch.cat((
			repr_p, repr_h,
			torch.abs(repr_p - repr_h),
			repr_p * repr_h
		),1)

		# Run via NN
		out = F.relu(self.hidden1(feedforward_input))
		out = self.dropout1(out)
		out = F.relu(self.hidden2(out))
		out = self.dropout2(out)
		out = self.hidden3(out)

		return F.softmax(out)


class SentReprDataset(Dataset):
	'''
	Dataset managing samples consisting of sentence representations
	'''

	def __init__(self, samples):
		self.samples = [(
				torch.from_numpy(rep_p).float(),
				torch.from_numpy(rep_h).float(),
				lbl
			) for rep_p, rep_h, lbl in samples]

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		return self.samples[idx]

def train(data_train, data_dev, iterations, dropout, lr, dim_hidden, name, batch_size=32, dim_out=3, validate_after=10):

	model_name = name + '.simplemodel'

	def evaluate(data, classifier):
		classifier.eval()

		correct = 0
		total = 0

		for p, h, lbl in data:

			total += p.size()[0]

			var_p = m.cuda_wrap(autograd.Variable(p))
			var_h = m.cuda_wrap(autograd.Variable(h))
			var_lbl = m.cuda_wrap(autograd.Variable(lbl))

			prediction = classifier(var_p, var_h)
			_, predicted_idx = torch.max(prediction, dim=1)
			correct += torch.sum(torch.eq(var_lbl, predicted_idx)).data[0]
		
		classifier.train()
		return correct / total	

	# For training
	data_loader_train = DataLoader(data_train, drop_last=True, batch_size=batch_size, shuffle=True)
	
	# For evaluation
	data_loader_dev_eval = DataLoader(data_dev, drop_last=False, batch_size=batch_size, shuffle=False)
	data_loader_train_eval = DataLoader(data_train, drop_last=False, batch_size=batch_size, shuffle=False)

	# Create and train model
	input_dim = data_train[0][0].size()[0]
	classifier = m.cuda_wrap(ReprClassifier(input_dim, dim_hidden, dim_out, dropout))

	start_lr = lr
	until_validation = 0
	samples_seen = 0
	optimizer = optim.Adam(classifier.parameters(), lr=lr)
	best_dev_acc = 0
	best_train_acc = 0
	for iteration in range(iterations):
		print('Running iteration', iteration + 1)
		for repr_p, repr_h, lbls in data_loader_train:
			
			until_validation -= repr_p.size()[0]
			samples_seen += repr_p.size()[0]

			# undo previous gradients
			classifier.zero_grad()
			optimizer.zero_grad()

			# Wrap in variables
			var_p = m.cuda_wrap(autograd.Variable(repr_p))
			var_h = m.cuda_wrap(autograd.Variable(repr_h))
			var_lbl = m.cuda_wrap(autograd.Variable(lbls))

			# predict and calculate error
			prediction = classifier(var_p, var_h)
			mean_loss = F.cross_entropy(prediction, var_lbl)
			mean_loss.backward()
			optimizer.step()

			if until_validation <= 0:
				until_validation = validate_after
				# evaluate
				print('After seeing', samples_seen, 'samples:')
				
				train_acc = evaluate(data_loader_train_eval, classifier)
				dev_acc = evaluate(data_loader_dev_eval, classifier)

				if dev_acc > best_dev_acc:
					print('Storing new best model:', model_name)
					best_train_acc = train_acc
					best_dev_acc = dev_acc
					torch.save(classifier.state_dict(), 'models/' + model_name)

				print('Accuracy on train:', train_acc)
				print('Accuracy on dev:', dev_acc)


		# Half lr decay
		decay = iteration // 2
		lr = start_lr / (2 ** decay)  
		for pg in optimizer.param_groups:
			pg['lr'] = lr

	print('Done')
	print('best model:', best_train_acc, best_dev_acc)

def load_data(folder, data_type, dimensions):
	'''
	This uses pre-saved sentence representations, genreated from the tool:

	$ python eval_twist.py <model> <data_train> <data_dev> <statpath> misclassified_sents_mf

	:param folder 		folder where the preprocessed sentence representations are stored
	:param data_type 	either train/dev
	'''

	pre = folder + 'invert_4m4f_' + data_type
	file_names = [
		pre + '-correct_correct.txt',
		pre + '-correct_incorrect.txt',
		pre + '-incorrect_correct.txt',
		pre + '-incorrect_incorrect.txt'
	]

	def chunker(seq, size):
		return (seq[pos:pos + size] for pos in range(0, len(seq), size))

	all_data_raw = []
	dt = np.dtype(float)
	for file in file_names:
		with open(file) as f_in:
			data = [(
				np.take(np.asarray(d[2].strip().split(' '), dtype=dt), dimensions),
				np.take(np.asarray(d[5].strip().split(' '), dtype=dt), dimensions), 
				int(d[6].strip().split(' ')[0])
			) for d in chunker(f_in.readlines(), 9)]
		all_data_raw += data

	return SentReprDataset(all_data_raw)


SD_DIMENSIONS = [757, 258, 35, 713, 602, 630, 199, 1787, 1730, 280, 89, 845, 1840, 311, 825, 809, 250, 184, 698, 1480, 1232, 817, 1987, 2007, 609, 1311, 683, 186, 1381, 1170, 107, 587, 818, 1878, 377, 475, 527, 1341, 1663, 480, 262, 284, 274, 1949, 1268, 1890, 778, 813, 1893, 977, 1466, 1307, 1251, 1608, 513, 2037, 1903, 804, 646, 1822, 1963, 1049, 140, 229, 731, 263, 1882, 1634, 965, 980, 742, 495, 1627, 188, 266, 1745, 1520, 1657, 45, 1156, 1393, 1032, 1238, 2044, 756, 1175, 882, 2029, 1223, 1370, 1174, 2020, 957, 1849, 86, 1687, 542, 1606, 1631, 1812, 1329, 350, 1276, 826, 1082, 1392, 138, 384, 485, 1088, 1282, 2001, 1449, 486, 1343, 1604, 1339, 1239, 1425, 1257, 1376, 1789, 1902, 1193, 1111, 1138, 952, 1695, 707, 1658, 569, 1979, 370, 1539, 1607, 363, 1901, 1327, 1144, 1926, 1877, 1484, 956, 1578, 1869, 1850, 1190, 1836, 1361, 798, 252, 842, 2042, 119, 773, 1094, 195, 395, 846, 1068, 304, 2015, 828, 878, 1624, 1796, 1231, 629, 668, 872, 62, 1131, 98, 1833, 690, 1805, 2017, 1842, 1301, 1794, 830, 1294, 1451, 1542, 1587, 369, 758, 1045, 1859, 1162, 234, 2014, 29, 1621, 134, 1414, 375, 1793, 193, 1073, 228, 1216, 529, 1852, 1464, 821, 1195, 1004, 139, 1672, 175, 2023, 1986, 538, 222, 1941, 970, 1512, 2016, 1186, 914, 1026, 124, 1884, 1147, 1142, 1821, 1858, 615, 402, 1062, 1513, 449, 1978, 616, 1125, 1955, 577, 1650, 2012, 9, 1442, 406, 1626, 722, 1113, 1437, 1133, 784, 4, 1728, 1824, 592, 15, 1596, 718, 1355, 1318, 1234, 43, 1737, 1387, 978, 1034, 2038, 1661, 1263, 226, 1853, 1143, 1888, 1220, 467, 33, 1014, 925, 362, 1211, 115, 2019, 719, 1629, 292, 413, 1819, 1603, 1990, 1868, 1396, 1610, 1863, 2047, 58, 1160, 497, 1058, 1847, 390, 458, 1516, 407, 1546, 1549, 1871, 973, 949, 1814, 198, 1242, 1421, 644, 1934, 1075, 1202, 1597, 100, 1298, 117, 472, 1617, 338, 1731, 1145, 1565, 463, 2000, 1454, 1727, 317, 1816, 1802, 1527, 575, 1404, 396, 1399, 909, 1067, 554, 177, 1486, 2005, 1981, 1785, 456, 966, 1371, 1319, 1643, 1286, 1673, 2025, 1753, 1838, 1194, 277, 622, 579, 1887, 297, 1183, 855, 1154, 607, 1585, 44, 1262, 265, 941, 2033, 1260, 1313, 1005, 983, 1119, 1765, 78, 1572, 1723, 621, 1827, 1755, 936, 998, 507, 1418, 660, 1974, 1965, 1826, 1544, 1947, 1227, 2013, 1105, 403, 1555, 1584, 1243, 1331, 1754, 366, 1670, 1592, 1570, 1770, 710, 5, 2030, 1576, 1198, 1741, 624, 1482, 1021, 1196, 399, 332, 1924, 727, 1639, 894, 136, 1865, 1218, 1429, 1116, 686, 811, 1167, 2036, 251, 1589, 1096, 1681, 201, 1932, 1594, 1885, 440, 858, 581, 477, 1471, 1906, 570, 101, 82, 1460, 1104, 1928, 171, 816, 19, 1375, 1493, 106, 852, 834, 1952, 1505, 230, 868, 820, 389, 1936, 1253, 2011, 556, 1055, 1076, 1641, 908, 1132, 1101, 6, 919, 1790, 1340, 1309, 767, 1324, 329, 1835, 1015, 1619, 1855, 1128, 215, 1697, 1729, 1362, 1684, 1408, 1078, 937, 1567, 939, 313, 1899, 689, 1976, 382, 289, 844, 3, 1237, 1018, 1937, 1201, 409, 1628, 2034, 242, 1720, 597, 1969, 1069, 1705, 534, 785, 1540, 1423, 1409, 1664, 103, 803, 1861, 1514, 1157, 340, 1189, 548, 1398, 1786, 1895, 72, 1769, 1308, 703, 430, 217, 94, 1590, 979, 1200, 1458, 1846, 361, 214, 1044, 1199, 1434, 1035, 1960, 948, 1415, 1074, 1296, 1529, 560, 1685, 1994, 1547, 643, 1803, 1148, 1575, 1346, 105, 558, 595, 1758, 611, 1900, 1224, 1711, 282, 639, 261, 1509, 1031, 1929, 1450, 1683, 83, 1334, 1897, 1783, 522, 1457, 777, 788, 1405, 1469, 1306, 1106, 1688, 1973, 1588, 1061, 182, 126, 1401, 1800, 1839, 237, 1605, 823, 704, 1719, 166, 374, 576, 221, 1256, 1270, 167, 1528, 1768, 1225, 1809, 1774, 755, 371, 1908, 1441, 770, 892, 1749, 574, 376, 271, 1958, 1149, 1517, 1917, 1693, 328, 1716, 498, 1151, 1548, 1613, 839, 165, 18, 1582, 1508, 249, 1036, 143, 1317, 59, 1518, 360, 1725, 1463, 1158, 468, 1320, 408, 678, 1182, 1690, 801, 2022, 1820, 953, 1250, 605, 1084, 760, 1534, 705, 87, 680, 988, 1310, 702, 1953, 434, 283, 1538, 1615, 260, 1008, 922, 519, 257, 1636, 897, 1675, 1909, 1692, 342, 642, 701, 958, 287, 1982, 1492, 1851, 1956, 1430, 1136, 1109, 57, 724, 1642, 69, 656, 355, 1776, 223, 848, 812, 1472, 1126, 590, 90, 276, 1536, 30, 771, 1652, 793, 1823, 1915, 1432, 1983, 943, 1837, 1130, 1944, 1530, 108, 99, 708, 1077, 1407, 1206, 163, 631, 734, 401, 1056, 606, 1345, 780, 827, 986, 268, 759, 1278, 1051, 1666, 1337, 441, 932, 231, 906, 1235, 1857, 876, 1886, 1795, 1394, 404, 1244, 751, 1531, 1391, 53, 1267, 1563, 1382, 1050, 1739, 218, 1115, 987, 787, 1275, 967, 1475, 673, 1618, 1566, 1207, 1070, 132, 450, 598, 1269, 658, 1372, 1040, 1474, 135, 1558, 1436, 1205, 454, 961, 279, 40, 1140, 1989, 191, 1448, 1496, 509, 728, 196, 436, 302, 944, 1217, 1694, 267, 1097, 1648, 1651, 512, 696, 649, 320, 672, 172, 508, 1354, 7, 470, 457, 1778, 1433, 1134, 1085, 1586, 79, 1264, 1356, 566, 946, 723, 528, 1813, 1524, 1550, 1230, 651, 1287, 1184, 1252, 383, 1726, 626, 1775, 452, 870, 1991, 123, 1222, 530, 347, 541, 1721, 379, 281, 653, 1510, 1121, 1702, 981, 1197, 1038, 1599, 1041, 888, 151, 1734, 1025, 768, 684, 358, 1465, 63, 1788, 851, 1645, 847, 388, 1013, 836, 934, 521, 883, 2045, 1703, 562, 693, 514, 185, 675, 1722, 1736, 142, 1988, 1701, 1489, 1473, 26, 822, 885, 638, 1993, 691, 131, 832, 1360, 290, 2, 582, 294, 754, 116, 572, 544, 510, 1420, 1898, 312, 211, 931, 445, 893, 1098, 1188, 1880, 545, 471, 1761, 918, 1255, 1181, 1556, 1123, 1129, 1092, 1299, 1876, 27, 325, 552, 889, 1048, 301, 208, 491, 158, 1713, 761, 429, 1818, 80, 368, 650, 933, 599, 240, 802, 1233, 667, 1330, 1948, 1108, 938, 414, 1782, 469, 1997, 910, 10, 1654, 1667, 762, 666, 1476, 505, 664, 183, 525, 1100, 1462, 1682, 1931, 157, 670, 1611, 16, 306, 1245, 426, 1945, 1999, 239, 2008, 857, 1219, 1874, 1746, 1326, 37, 1757, 1922, 738, 176, 1574, 466, 42, 295, 300, 729, 789, 955, 206, 47, 935, 1023, 1951, 610, 1266, 2010, 1598, 88, 632, 1740, 911, 585, 380, 1614, 1152, 1312, 1889, 1678, 451, 1444, 517, 1784, 1490, 744, 950, 665, 891, 159, 36, 819, 652, 1977, 589, 394, 1379, 1879, 1560, 1773, 1996, 700, 235, 227, 1483, 1236, 353, 628, 253, 997, 1959, 1210, 307, 1295, 97, 92, 487, 1777, 596, 1455, 2024, 763, 1043, 895, 1117, 418, 1203, 349, 324, 410, 1124, 1831, 2026, 721, 432, 364, 536, 619, 321, 153, 716, 1305, 1141, 254, 1029, 1288, 1919, 1350, 241, 1581, 504, 1638, 149, 1699, 877, 603, 1087, 255, 1146, 1293, 783, 1551, 291, 81, 31, 2021, 1691, 782, 1710, 499, 567, 1279, 224, 337, 1066, 1452, 1593, 1478, 2040, 795, 928, 1179, 862, 749, 113, 464, 1030, 398, 1759, 225, 563, 1709, 1891, 446, 1961, 1714, 397, 333, 766, 1435, 28, 1637, 1780, 1248, 1060, 1187, 1804, 748, 1860, 1093, 55, 303, 697, 1653, 64, 1209, 995, 1905, 533, 502, 1925, 1443, 1495, 1363, 179, 900, 1808, 161, 1166, 61, 483, 838, 435, 1834, 1655, 1099, 661, 216, 205, 323, 500, 1950, 976, 286, 814, 1020, 1017, 1750, 905, 1416, 1011, 963, 1395, 775, 1971, 1258, 120, 1366, 1336, 1743, 1706, 1647, 614, 1240, 326, 1732, 423, 539, 1303, 1564, 1630, 160, 1063, 964, 503, 860, 663, 49, 736, 1445, 1033, 921, 612, 625, 1573, 1071, 1957, 1497, 1862, 1921, 1163, 1715, 1913, 1562, 54, 835, 1177, 1037, 310, 1291, 391, 1939, 110, 269, 60, 1342, 1427, 511, 531, 1708, 1747, 1357, 1625, 400, 1259, 1459, 1867, 1002, 96, 455, 535, 584, 982, 453, 1347, 315, 1561, 1552, 730, 1704, 1525, 687, 601, 1281, 637, 244, 694, 1698, 709, 586, 2028, 0, 1229, 417, 1799, 1507, 1000, 506, 712, 1810, 1553, 484, 20, 220, 926, 1781, 431, 1856, 1277, 810, 424, 930, 194, 420, 679, 1089, 444, 493, 840, 118, 1498, 833, 1388, 481, 1577, 837, 1214, 714, 1365, 345, 173, 901, 121, 1677, 256, 1328, 794, 405, 1503, 210, 385, 330, 1081, 1185, 1501, 1411, 1052, 1118, 1254, 378, 411, 645, 1557, 147, 677, 873, 127, 765, 1453, 791, 1844, 520, 114, 415, 711, 1164, 1228, 1246, 264, 951, 1173, 1016, 1477, 1386, 1091, 148, 808, 1762, 189, 807, 288, 1533, 1674, 1439, 1828, 318, 25, 1412, 461, 1995, 1623, 1633, 1751, 1302, 1368, 1024, 439, 95, 747, 146, 17, 1504, 565, 1352, 532, 1006, 1649, 1875, 786, 1841, 1383, 170, 1272, 524, 853, 1616, 1738, 1764, 1535, 1541, 1848, 1825, 1403, 867, 850, 741, 1545, 202, 699, 1696, 1172, 73, 604, 1864, 1511, 1380, 752, 1039, 1600, 76, 1261, 1012, 1622, 232, 874, 779, 849, 1461, 1022, 314, 573, 1417, 863, 322, 1086, 662, 1153, 972, 824, 1374, 2002, 1120, 316, 1114, 190, 52, 341, 1676, 2041, 246, 365, 129, 133, 692, 954, 580, 2004, 685, 843, 357, 516, 248, 960, 923, 523, 1669, 2039, 1689, 482, 75, 1872, 1717, 1817, 571, 1656, 459, 1137, 77, 273, 67, 1159, 555, 647, 387, 359, 733, 1918, 1440, 299, 156, 1155, 1894, 447, 476, 695, 674, 164, 1419, 181, 275, 546, 1771, 887, 1602, 1, 769, 648, 102, 1204, 2046, 635, 2006, 781, 1059, 543, 1107, 1494, 1797, 144, 1304, 753, 796, 654, 1064, 272, 1470, 1400, 884, 1521, 128, 613, 1832, 1646, 70, 1830, 968, 1779, 1438, 896, 169, 200, 213, 51, 1968, 564, 1767, 490, 1916, 681, 212, 1359, 659, 1332, 1854, 1559, 1942, 1752, 1072, 1843, 1506, 425, 356, 655, 293, 985, 1980, 869, 1080, 422, 1090, 309, 1323, 236, 32, 594, 1122, 561, 38, 65, 924, 1892, 91, 947, 881, 1522, 797, 197, 1920, 657, 999, 732, 1479, 1998, 886, 1456, 989, 416, 8, 578, 1712, 1912, 1870, 386, 1724, 48, 344, 1384, 427, 1057, 715, 962, 1406, 726, 641, 720, 1265, 829, 1169, 327, 725, 259, 492, 537, 927, 1665, 550, 549, 23, 1351, 419, 1297, 792, 1284, 305, 1766, 339, 442, 915, 348, 2027, 187, 496, 1881, 1292, 209, 1241, 1353, 774, 2003, 1315, 488, 871, 1569, 776, 2018, 815, 1791, 111, 1344, 438, 1221, 739, 109, 620, 1349, 93, 903, 1938, 859, 540, 996, 192, 1192, 1718, 354, 745, 991, 1273, 1322, 125, 247, 1335, 66, 1807, 12, 1935, 841, 984, 1422, 717, 740, 945, 1984, 1744, 21, 907, 238, 806, 433, 56, 608, 854, 636, 478, 1467, 1632, 1283, 1042, 1325, 331, 1053, 1811, 1176, 1967, 141, 1373, 270, 515, 1742, 1543, 1249, 308, 1378, 41, 152, 85, 865, 393, 1680, 913, 1390, 1635, 805, 600, 1428, 68, 1369, 1668, 669, 479, 1161, 682, 1845, 593, 1247, 1333, 1001, 735, 2043, 1285, 912, 799, 285, 474, 1112, 1009, 743, 296, 1385, 1671, 1168, 489, 1700, 1914, 1180, 800, 298, 1178, 1103, 1321, 583, 591, 1491, 1554, 74, 627, 975, 916, 671, 1946, 993, 1487, 412, 112, 1756, 1644, 917, 1954, 104, 929, 145, 890, 1348, 588, 34, 547, 71, 381, 278, 899, 1568, 335, 392, 1413, 1792, 994, 1499, 346, 1907, 1010, 1970, 1829, 1975, 1735, 1274, 1364, 1127, 1933, 460, 1679, 336, 1659, 1815, 334, 1733, 640, 974, 150, 1515, 1358, 1431, 2032, 688, 1930, 1620, 1446, 1962, 1526, 623, 1966, 1488, 864, 990, 617, 551, 2031, 1502, 676, 1595, 465, 154, 1686, 1426, 1866, 155, 1367, 1389, 233, 1910, 372, 1972, 207, 319, 22, 1171, 1110, 880, 14, 473, 13, 559, 1054, 1896, 902, 1150, 879, 1226, 861, 618, 1215, 1289, 352, 1447, 764, 1019, 1992, 245, 1290, 568, 494, 1707, 1065, 443, 343, 1640, 130, 1424, 122, 1280, 959, 1760, 1985, 1208, 1314, 1212, 1095, 706, 526, 178, 1904, 1806, 831, 1579, 1883, 1601, 746, 750, 448, 1591, 633, 162, 501, 1300, 1402, 942, 1772, 1102, 856, 1612, 1079, 11, 174, 428, 557, 1923, 1580, 1083, 203, 1410, 137, 219, 969, 46, 2009, 1027, 898, 462, 1377, 737, 1003, 1165, 1964, 1519, 1481, 1609, 1571, 1873, 1940, 1927, 50, 1532, 351, 1047, 1191, 1213, 1660, 180, 904, 168, 1139, 1583, 553, 518, 1271, 1468, 1046, 1537, 866, 421, 1028, 1135, 1801, 1911, 1763, 971, 1397, 39, 1748, 367, 992, 2035, 1943, 1523, 1798, 940, 24, 1338, 790, 1500, 243, 1007, 1485, 875, 634, 1316, 772, 204, 1662, 84, 373, 920, 437]

def main():
	torch.manual_seed(6)
	args = docopt("""Train a model based on existing sentence representation. 
		Only use a subset of the existing sentence representations.

		exact: specify dimensions seperated by whitespace
		top:   specify amount of top dimensions by SD

	Usage:
    	sent_representation_classify.py exact <folder> <dimensions> --lr=<lr> --n=<name> --drp=<dropout> --dh=<dim_hidden> --b=<batch_size> --v=<validate_after>
    	sent_representation_classify.py top <folder> <dimensions> --lr=<lr> --n=<name> --drp=<dropout> --dh=<dim_hidden> --b=<batch_size> --v=<validate_after>
	""")

	folder = args['<folder>']
	if args['exact']:
		print('exact')
		dimensions = np.asarray(args['<dimensions>'].strip().split(' '), dtype=int)
	else:
		print('top')
		dimensions = np.asarray(SD_DIMENSIONS[:int(args['<dimensions>'].strip())])

	data_train = load_data(folder, 'train', dimensions)
	data_dev = load_data(folder, 'dev', dimensions)

	name = args['--n']
	dropout = float(args['--drp'])
	dim_hidden = int(args['--dh'])
	batch_size = int(args['--b'])
	lr = float(args['--lr'])
	validate_after = int(args['--v'])

	name = '_'.join([name, str(dropout), str(dim_hidden), str(batch_size), str(lr)])
	train(data_train, data_dev, iterations=5, dropout=dropout, lr=lr, dim_hidden=dim_hidden, name=name, batch_size=batch_size, validate_after=validate_after, dim_out=3)

if __name__ == '__main__':
	main()
