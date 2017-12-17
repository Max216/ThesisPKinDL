import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
import numpy as np
import random

from docopt import docopt


SD_DIMENSIONS = [757, 258, 35, 713, 602, 1232, 630, 199, 1787, 280, 1730, 89, 845, 311, 1840, 250, 809, 1987, 698, 825, 184, 817, 609, 377, 1480, 1381, 1393, 1850, 2007, 1170, 107, 587, 1311, 186, 527, 683, 475, 1878, 646, 480, 262, 284, 62, 818, 513, 1877, 1663, 1341, 252, 1890, 977, 274, 1466, 1893, 1882, 1822, 1268, 229, 4, 804, 1634, 1949, 1903, 1963, 1608, 263, 1175, 758, 813, 1307, 1251, 778, 1049, 140, 1657, 1355, 742, 2037, 495, 2001, 980, 882, 1484, 1238, 266, 965, 1631, 1156, 188, 86, 1282, 1520, 45, 731, 1223, 350, 1174, 1789, 1687, 1343, 1088, 2029, 1032, 2044, 2020, 1257, 1193, 1392, 1695, 1849, 1144, 957, 756, 1082, 1539, 119, 1627, 1607, 1339, 1370, 370, 1449, 826, 1812, 1239, 1979, 1329, 1986, 486, 363, 304, 1794, 384, 1376, 395, 1745, 1276, 1901, 1327, 707, 1138, 1926, 1606, 542, 1624, 15, 1301, 1111, 485, 956, 1578, 369, 138, 952, 134, 690, 1361, 1869, 798, 1004, 1658, 2015, 2042, 1805, 919, 842, 1231, 668, 569, 1190, 1833, 1294, 1727, 139, 878, 1425, 2014, 1068, 949, 1836, 1902, 1859, 1604, 1842, 2017, 2023, 1621, 1094, 1216, 228, 467, 872, 773, 830, 1131, 1414, 98, 828, 1681, 1796, 577, 1142, 292, 821, 1034, 1073, 2012, 1793, 1852, 1062, 1661, 195, 234, 2016, 529, 1819, 616, 846, 1542, 784, 29, 1821, 375, 579, 1045, 193, 1587, 33, 106, 1396, 914, 1160, 1731, 1451, 2038, 1847, 1868, 1113, 1075, 124, 538, 9, 1195, 58, 1978, 402, 1262, 1513, 1955, 1626, 978, 406, 413, 115, 1941, 1464, 1162, 1824, 939, 1147, 1442, 497, 629, 1125, 1629, 615, 1512, 941, 449, 1026, 1143, 1186, 1298, 1672, 265, 1858, 1220, 1990, 1670, 1650, 970, 1437, 1202, 1133, 592, 722, 1387, 1888, 719, 2019, 1234, 1597, 1263, 1884, 222, 1318, 1399, 1863, 226, 1527, 1058, 1728, 1154, 1021, 2000, 1594, 1546, 1885, 43, 554, 1737, 463, 575, 19, 607, 925, 1934, 973, 909, 1565, 1853, 1585, 407, 366, 458, 2047, 1603, 1242, 472, 1969, 1981, 1770, 1516, 718, 297, 1887, 1549, 329, 338, 1729, 1785, 1324, 1596, 1319, 1105, 1145, 1802, 1211, 1871, 1741, 317, 1486, 1505, 175, 1838, 390, 2005, 936, 100, 477, 1643, 1544, 117, 1119, 1371, 1753, 1641, 1404, 1754, 1286, 362, 1617, 1198, 1482, 1331, 1684, 177, 1313, 1723, 966, 534, 622, 198, 507, 644, 2033, 1673, 1683, 1418, 78, 1619, 983, 1069, 1826, 1421, 1194, 1610, 1584, 1937, 1067, 1256, 1567, 1005, 834, 868, 1816, 1260, 1572, 251, 894, 855, 1018, 811, 1458, 727, 396, 1167, 403, 456, 2030, 1296, 1755, 2013, 2036, 1570, 1765, 230, 1429, 6, 399, 1014, 1471, 277, 686, 621, 1243, 1865, 5, 1835, 1855, 1932, 1116, 556, 1906, 710, 1055, 998, 1639, 1555, 2025, 1936, 440, 2011, 44, 1196, 1227, 332, 1576, 83, 816, 1814, 1183, 1589, 660, 1974, 570, 1454, 1218, 788, 374, 858, 1104, 1697, 242, 1158, 1783, 1952, 1590, 215, 1928, 1965, 1375, 1827, 201, 1592, 624, 1947, 908, 2034, 101, 171, 581, 1132, 409, 767, 1408, 1253, 1924, 1769, 94, 820, 1790, 1460, 1128, 1199, 1493, 313, 1096, 1899, 1074, 1711, 689, 1076, 1509, 1953, 777, 82, 382, 1309, 844, 1547, 59, 289, 1237, 1423, 1101, 223, 1958, 389, 1786, 1540, 852, 1200, 1148, 1409, 643, 937, 1201, 1960, 979, 1031, 1044, 261, 1994, 1517, 103, 1758, 1976, 1078, 1575, 597, 1628, 3, 361, 785, 1398, 1035, 1720, 72, 548, 771, 1688, 136, 1189, 1895, 948, 1362, 1015, 1685, 1861, 1705, 340, 1795, 639, 1340, 214, 182, 1514, 703, 1839, 1434, 1529, 1334, 595, 1415, 1157, 1917, 1308, 522, 1820, 1900, 1310, 1457, 401, 1306, 430, 167, 576, 1664, 1534, 1346, 1909, 166, 1588, 217, 1929, 1719, 1405, 1803, 1725, 1106, 1605, 1224, 1846, 519, 574, 1774, 704, 560, 105, 221, 755, 953, 126, 1450, 1401, 958, 371, 1908, 1508, 1809, 1469, 801, 1061, 803, 1548, 558, 823, 1225, 1956, 1582, 342, 1270, 328, 1897, 1768, 237, 282, 376, 1149, 1800, 1528, 87, 1463, 892, 780, 642, 257, 1973, 1441, 468, 986, 1693, 1776, 605, 1716, 1036, 1250, 249, 1182, 1613, 1317, 760, 734, 283, 498, 1915, 1538, 271, 988, 839, 770, 702, 611, 1618, 143, 1008, 1320, 724, 1749, 705, 1690, 163, 678, 165, 1077, 701, 680, 1430, 434, 1536, 360, 1692, 1492, 656, 1636, 897, 922, 1982, 708, 276, 1675, 69, 1130, 1851, 18, 1615, 1235, 2022, 1244, 441, 1518, 1084, 759, 1407, 1136, 1206, 267, 943, 57, 1126, 260, 1432, 90, 1275, 1652, 1857, 287, 848, 1472, 1944, 590, 1151, 108, 355, 631, 1823, 793, 408, 1642, 1694, 1056, 658, 1345, 1115, 932, 1109, 30, 933, 1337, 135, 876, 1983, 1666, 812, 1278, 673, 827, 906, 1558, 1372, 1051, 1530, 1531, 1886, 606, 404, 1474, 509, 99, 231, 1837, 1184, 320, 1448, 1991, 1050, 512, 1267, 218, 268, 1391, 1140, 302, 944, 1524, 1739, 598, 1070, 961, 751, 2, 1566, 1651, 987, 787, 1382, 1563, 1040, 450, 454, 1433, 1217, 53, 1207, 279, 1085, 436, 132, 1264, 1205, 599, 638, 1475, 1354, 1648, 649, 470, 172, 1097, 151, 1496, 294, 1394, 1586, 566, 1778, 1436, 508, 967, 1041, 1222, 1269, 946, 1813, 1287, 651, 7, 40, 142, 562, 79, 1726, 191, 582, 471, 379, 63, 383, 672, 1465, 1702, 196, 1356, 528, 290, 693, 1134, 931, 131, 728, 183, 1013, 541, 1989, 847, 1654, 457, 1473, 211, 723, 870, 530, 388, 1775, 116, 684, 1230, 653, 691, 851, 281, 981, 696, 626, 1038, 1988, 1599, 1993, 2045, 1299, 1550, 10, 1510, 1645, 123, 1360, 1252, 934, 1788, 325, 822, 1025, 1736, 312, 1721, 158, 1898, 675, 1420, 358, 452, 521, 1703, 26, 836, 1121, 1092, 768, 910, 157, 545, 1701, 1108, 664, 1098, 1880, 514, 208, 1757, 185, 918, 544, 893, 1129, 1556, 1948, 885, 1489, 1734, 888, 347, 754, 295, 505, 445, 1123, 1667, 1476, 938, 761, 239, 1181, 552, 832, 469, 1462, 368, 2008, 1048, 1255, 857, 1722, 1931, 176, 667, 1682, 1197, 240, 1691, 1233, 1611, 670, 666, 414, 883, 802, 300, 525, 1100, 1188, 650, 889, 36, 80, 1288, 429, 1818, 510, 27, 301, 1551, 1876, 37, 1997, 572, 1782, 1761, 517, 1330, 426, 762, 1945, 1236, 1245, 1713, 1999, 596, 963, 315, 1117, 610, 1312, 1614, 585, 2010, 619, 466, 1678, 1560, 16, 1746, 1326, 42, 1951, 1598, 88, 1874, 1023, 935, 1831, 2040, 499, 950, 97, 1922, 789, 1444, 1862, 729, 1152, 997, 1490, 700, 418, 92, 766, 353, 1699, 1219, 306, 652, 911, 380, 877, 206, 1740, 227, 744, 955, 665, 235, 1977, 632, 535, 1379, 1959, 324, 291, 891, 1203, 1889, 1996, 1759, 738, 432, 81, 253, 255, 349, 1879, 1066, 1266, 819, 536, 1455, 153, 394, 47, 1919, 1581, 451, 1637, 491, 159, 321, 1124, 603, 1303, 1063, 1593, 2026, 224, 1674, 976, 862, 307, 113, 487, 1087, 2024, 1773, 589, 1830, 149, 1750, 628, 1029, 895, 64, 749, 763, 1293, 419, 783, 364, 716, 1305, 410, 1093, 1416, 1146, 241, 995, 1210, 1784, 504, 435, 1860, 1961, 721, 748, 398, 1710, 563, 464, 61, 1638, 1043, 500, 1630, 337, 1655, 1030, 446, 303, 31, 1363, 1179, 28, 539, 1295, 1625, 1574, 900, 782, 502, 1561, 179, 120, 2021, 1240, 1064, 1780, 1804, 1141, 1483, 1229, 1258, 160, 1011, 1913, 838, 1709, 775, 533, 1777, 567, 814, 1867, 1808, 1834, 310, 216, 1925, 1002, 423, 1704, 1166, 1163, 1781, 1564, 1187, 1495, 1342, 860, 1497, 254, 1052, 1743, 1395, 1653, 736, 1328, 326, 55, 1443, 1060, 397, 1209, 1706, 1366, 1939, 1971, 1017, 1020, 0, 905, 511, 161, 1715, 645, 1573, 1714, 2002, 506, 60, 420, 54, 928, 687, 400, 1071, 232, 663, 1336, 1445, 205, 1553, 333, 1427, 323, 391, 586, 1291, 503, 964, 1037, 1248, 1647, 531, 1921, 1905, 1357, 661, 244, 741, 225, 1562, 1478, 275, 745, 483, 835, 1452, 625, 1279, 1698, 67, 194, 1995, 493, 671, 1708, 584, 127, 1950, 444, 709, 612, 286, 1033, 431, 1732, 921, 523, 1751, 1277, 1525, 1365, 49, 1347, 1799, 679, 1507, 96, 637, 614, 520, 795, 926, 1177, 1099, 1747, 1388, 220, 417, 1435, 1214, 481, 714, 453, 484, 712, 837, 118, 810, 1623, 210, 532, 692, 982, 1810, 424, 1557, 1006, 697, 415, 730, 455, 164, 1501, 1081, 601, 385, 405, 322, 1172, 20, 791, 269, 1511, 1957, 1090, 2028, 147, 694, 1616, 1016, 1228, 1459, 779, 1552, 256, 264, 807, 1173, 202, 1259, 1089, 189, 1891, 1503, 121, 1762, 1600, 1281, 711, 833, 808, 1411, 930, 378, 840, 1864, 1649, 1000, 873, 1633, 1535, 1577, 849, 2004, 951, 1412, 1856, 150, 1417, 752, 148, 1164, 288, 1828, 565, 524, 1738, 824, 843, 1246, 1696, 439, 1677, 1825, 1403, 1024, 1374, 1155, 1477, 345, 677, 1352, 411, 1461, 588, 1872, 314, 170, 765, 1118, 874, 114, 1159, 901, 330, 578, 674, 853, 786, 1764, 52, 1350, 1053, 739, 662, 461, 17, 555, 2041, 543, 1059, 604, 1439, 1107, 212, 613, 903, 70, 365, 1419, 1185, 133, 896, 2039, 1386, 1848, 110, 747, 1091, 1120, 173, 1272, 850, 359, 1894, 1380, 1541, 863, 190, 516, 387, 1254, 25, 887, 1453, 213, 1368, 2046, 273, 1918, 459, 769, 960, 685, 1669, 871, 156, 129, 1832, 1844, 341, 1676, 924, 482, 1656, 144, 76, 91, 1204, 425, 73, 141, 1440, 1383, 1559, 699, 695, 608, 947, 75, 65, 1602, 659, 571, 246, 476, 1875, 1545, 923, 580, 357, 95, 635, 1, 1012, 655, 272, 128, 546, 681, 1843, 1870, 884, 236, 1622, 490, 1717, 794, 647, 564, 2003, 1646, 1080, 1942, 318, 1265, 1533, 867, 1373, 422, 929, 797, 1039, 447, 968, 1359, 1771, 181, 1506, 1022, 200, 51, 1504, 1332, 1689, 984, 869, 989, 1438, 781, 299, 594, 899, 309, 1817, 356, 881, 1153, 427, 1752, 1498, 726, 999, 496, 152, 972, 1797, 1779, 954, 1479, 620, 573, 1521, 1892, 344, 1169, 753, 1086, 792, 1302, 654, 561, 1767, 125, 169, 733, 1384, 2006, 648, 32, 248, 583, 1494, 829, 416, 796, 1841, 293, 1854, 641, 1968, 720, 1766, 192, 111, 1249, 1916, 1680, 1284, 1938, 991, 38, 1297, 1470, 1072, 1772, 319, 1724, 348, 732, 1390, 1935, 962, 259, 1920, 776, 706, 1744, 1137, 21, 927, 488, 386, 1718, 392, 764, 1522, 1057, 1042, 657, 247, 1273, 197, 774, 815, 669, 209, 77, 1632, 162, 1791, 473, 1912, 1325, 715, 298, 1083, 886, 725, 102, 1261, 145, 915, 12, 1241, 2018, 1456, 540, 316, 1406, 1304, 8, 737, 433, 1967, 627, 913, 442, 1914, 1712, 1351, 1349, 56, 549, 1635, 104, 71, 912, 478, 1315, 1283, 1369, 985, 1467, 1980, 48, 1247, 2027, 327, 93, 591, 1954, 550, 1543, 146, 1122, 474, 22, 907, 717, 438, 1112, 841, 917, 1992, 1114, 339, 1292, 1970, 1323, 890, 1335, 1176, 537, 735, 34, 676, 859, 1668, 1221, 187, 68, 1385, 996, 1009, 1665, 865, 1285, 1274, 1499, 1422, 1829, 278, 14, 331, 636, 1010, 372, 238, 354, 1333, 975, 1322, 448, 682, 492, 66, 1700, 623, 412, 866, 856, 270, 1353, 740, 945, 593, 74, 1644, 902, 85, 600, 296, 1502, 112, 1428, 1998, 990, 743, 1192, 2043, 1907, 994, 1321, 1946, 393, 1554, 1595, 1966, 23, 1811, 1367, 1103, 109, 352, 13, 806, 1526, 1487, 1845, 305, 1568, 1883, 1910, 50, 381, 1079, 854, 916, 334, 1742, 346, 1180, 880, 1168, 1733, 1972, 1515, 1400, 1413, 1933, 460, 618, 1735, 1171, 462, 1640, 245, 1756, 1300, 547, 559, 1289, 1620, 336, 489, 864, 688, 942, 285, 800, 1447, 992, 1344, 1896, 1110, 1348, 1424, 41, 1212, 1431, 2031, 805, 1569, 1659, 746, 1161, 1226, 203, 308, 1054, 1127, 154, 1686, 1102, 875, 1410, 993, 479, 1881, 1314, 1001, 428, 155, 515, 526, 1208, 861, 1532, 551, 335, 1792, 1748, 1930, 501, 122, 1985, 1962, 1866, 1807, 640, 1378, 1139, 2032, 568, 207, 1984, 1815, 1491, 1150, 1927, 443, 617, 178, 343, 1488, 1019, 1178, 1446, 494, 1964, 465, 1679, 1290, 1975, 831, 1612, 130, 898, 974, 1316, 1426, 1923, 1806, 1377, 168, 1389, 1007, 1364, 11, 799, 1028, 1065, 243, 959, 1671, 1904, 1397, 1481, 219, 2009, 137, 1135, 1707, 971, 750, 1468, 1191, 518, 1580, 46, 557, 1095, 1798, 969, 1165, 1940, 1271, 1601, 1591, 1579, 174, 1215, 1047, 1943, 790, 1801, 1763, 633, 1402, 940, 39, 180, 634, 1280, 1358, 373, 1046, 1519, 1873, 1760, 24, 1660, 1027, 2035, 904, 1003, 1537, 879, 351, 1609, 553, 421, 1500, 1523, 1485, 1338, 1583, 233, 204, 1571, 84, 772, 1911, 367, 1213, 1662, 437, 920]
LBL_NOT_DATA = -5000
color_palette = ['#e6194b','#3cb44b','#ffe119','#0082c8','#f58231','#911eb4','#46f0f0','#000000','#f032e6','#d2f53c', '#fabebe','#008080','#aa6e28','#800000','#808000','#000080','#808080','#e6beff','#aaffc3','#fffac8','#ff0000','#00ff00','#0000ff','#ff00ff','#234567','#00ffff']


def print_sd_rank(params):
	folder = params['<folder>']
	pre = folder + 'invert_4m4f_train'
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
			data = [(d[0], d[1], d[2], d[3], d[4], d[5], d[6].strip().split(' ')) 
				for d in chunker(f_in.readlines(), 9)]
		all_data_raw += data

	# remember all representations
	unique_repr = []
	used_sents = set()
	while len(all_data_raw) > 0:
		current = all_data_raw.pop()
		if current[0] not in used_sents:
			used_sents.add(current[0])
			unique_repr.append(np.asarray(current[2].strip().split(' '), dtype=float))
		if current[3] not in used_sents:
			used_sents.add(current[3])
			unique_repr.append(np.asarray(current[5].strip().split(' '), dtype=float))
	repr_matrix = np.asmatrix(unique_repr)
	sd = np.asarray(np.std(repr_matrix, axis=0)).flatten()
	sd = sorted([(dim, s) for dim, s in enumerate(sd)], key=lambda x: -x[-1])
	print([dim for dim, _  in sd])

def generate_data(params):
	folder = params['<folder>']
	pre = folder + 'invert_4m4f_train'
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
			data = [(d[0], d[1], d[2], d[3], d[4], d[5], d[6].strip().split(' ')) 
				for d in chunker(f_in.readlines(), 9)]
		all_data_raw += data

	all_data_raw = [(p, p_act, p_rep, h, h_act, h_rep, int(lbl[0]), int(lbl[1])) 
		for p, p_act, p_rep, h, h_act, h_rep, lbl in all_data_raw]


	# Sort by premise
	all_data_raw = sorted(all_data_raw, key=lambda x: x[0])

	# divide samples into categories
	categories = dict()
	categories['correct'] = []
	categories['only_entailment_incorrect'] = []
	categories['only_contradiction_incorrect'] = []
	categories['only_neutral_incorrect'] = []

	incorrect_gold_lbl_to_cat = ['only_neutral_incorrect', 'only_contradiction_incorrect', 'only_entailment_incorrect']


	def find_category(samples):
		# must have all three labels
		if len(list(set([s[-2] for s in samples]))) != 3:
			return None

		incorrect_lbls = [s[-2] for s in samples if s[-2] != s[-1]]
		if len(incorrect_lbls) == 0:
			return 'correct'
		elif len(incorrect_lbls) == 1:
			return incorrect_gold_lbl_to_cat[incorrect_lbls[0]]
		else:
			return None

	while len(all_data_raw) > 0:
		# find subset having same premise
		last_idx = 1
		p  = all_data_raw[0][0]
		broke_out = False
		for i in range(1, len(all_data_raw)):
			if all_data_raw[i][0] != p:
				last_idx = i
				broke_out = True
				break

		if broke_out == False:
			last_idx = len(all_data_raw)

		sub_data = 	all_data_raw[:last_idx]
		del all_data_raw[:last_idx]
		
		# only use if exactly three samples
		if len(sub_data) == 3:
			cat = find_category(sub_data)
			if cat != None:
				# remember samples
				categories[cat].append(sub_data)

	# Select for usage
	def sample(data, amount):
		return [data[i] for i in sorted(random.sample(range(len(data)), amount))]

	sample_correct = sample(categories['correct'], 150)
	sample_only_entailment_incorrect = sample(categories['only_entailment_incorrect'], 50)
	sample_only_contradiction_incorrect = sample(categories['only_contradiction_incorrect'], 50)
	sample_only_neutral_incorrect = sample(categories['only_neutral_incorrect'], 50)

	# write to file
	name_out = folder + 'representation_samples_450_150_150_150.txt'
	idx_to_lbl = ['neutral', 'contradiction', 'entailment']
	with open(name_out, 'w') as f_out:
		# go via all categories
		for sample_set in [sample_correct, sample_only_entailment_incorrect, sample_only_contradiction_incorrect, sample_only_neutral_incorrect]:
			
			# via all groups sharing a premise
			for sample_group in sample_set:

				# via all samples within that group
				for p, p_act, p_rep, h, h_act, h_rep, lbl, lbl_predicted in sample_group:
					f_out.write(p)
					f_out.write(p_act)
					f_out.write(p_rep)
					f_out.write(h)
					f_out.write(h_act)
					f_out.write(h_rep)
					f_out.write(idx_to_lbl[lbl] + ' ' + idx_to_lbl[lbl_predicted] + '\n')

class Sample:
	'''
	A single sample consisting of premise and hypothesis together with 
	all activations and representations is stored compactly in this class.
	'''

	def __init__(self, p, p_act, p_rep, h, h_act, h_rep, lbl, predicted):
		self.p = p
		self.h = h
		self.p_act = p_act
		self.h_act = h_act
		self.p_rep = p_rep
		self.h_rep = h_rep
		self.lbl = lbl
		self.predicted = predicted
		self.dims = [i for i in range(len(p_act))]
		self.applied_filters = []

	def filter(self, filter_fn):
		'''
		Filters the data contained in this class using a filter_fn

		:param filter_fn 	must be a function taking a sample as input and returning 
							(name, dims). Only those dimensions in dims are kept.
		'''
		name, keep_dims = filter_fn(self)
		keep_dim_indizes = [i for i in range(len(self.dims)) if self.dims[i] in keep_dims]
		self.dims = np.take(self.dims, keep_dim_indizes)
		self.applied_filters.append(name)

	def get_applied_filters(self):
		'''
		:return the name appendix based on applied filters
		'''
		return '_'.join(self.applied_filters)

	def dimsize(self):
		return len(self.dims)


def load_sent(sample_idx, label, params):
	'''
	Load a singe sentence pair of premise and hypothesis.

	:param sample_idx 		Specify the index of the sample you want to examine.
							A sample consists of one premise with three different hypothesis and 
							three different labelclass Sample:

	:param label 			Take the sentence pair that has this gold label
	:params 				Additional parameters if filters are applied.
	'''
	path = './../analyses/representation_samples_450_150_150_150.txt'
	SAMPLE_SIZE = 7
	start_line = sample_idx * SAMPLE_SIZE * 3
	line_counter = 0

	# remember findings
	p = None
	p_act = None
	p_rep = None
	h = None
	h_act = None
	h_rep = None
	lbl = None
	predicted = None

	with open(path) as f_in:
		for line in f_in:

			# the correct set of sentence pairs started
			if  line_counter >= start_line:
				line = line.strip()
				rel_position = line_counter % SAMPLE_SIZE

				# remember infos
				if rel_position == 0:
					p = line.split(' ')
				elif rel_position == 1:
					p_act = np.asarray(line.split(' '), dtype=int)
				elif rel_position == 2:
					p_rep = np.asarray(line.split(' '), dtype=float)
				elif rel_position == 3:
					h = line.split(' ')
				elif rel_position == 4:
					h_act = np.asarray(line.split(' '), dtype=int)
				elif rel_position == 5:
					h_rep = np.asarray(line.split(' '), dtype=float)
				elif rel_position == 6:
					lbl = line.split(' ')[0]
					predicted = line.split(' ')[1]
					if lbl == label:
						# found correct pair
						break


			line_counter += 1

	sample = Sample(p, p_act, p_rep, h, h_act, h_rep, lbl, predicted)

	# only uses dimensions if in either p or h this threshold is reached.
	threshold_single = params.get('--t', None)
	# only uses dimensions if both values are >= threshold
	threshold_both = params.get('--tb', None)
	# only uses the dimensions within the ones with the most standard deviation
	top_sd = params.get('--tsd', None)
	
	# Filter functions
	def filter_threshold_single(sample):
		t = float(threshold_single)
		name = 't=' + threshold_single
		filtered_dims = [dim for dim in sample.dims if sample.p_rep[dim] >= t or sample.h_rep[dim] > t]
		return (name, filtered_dims)

	def filter_threshold_both(sample):
		tb = float(threshold_both)
		name = 'tb=' + threshold_both
		filtered_dims = [dim for dim in sample.dims if sample.p_rep[dim] >= tb and sample.h_rep[dim] > tb]
		return (name, filtered_dims)

	def filter_top_sd(sample):
		used_dims = SD_DIMENSIONS[:int(top_sd)]
		name='top=' + top_sd
		filtered_dims = [dim for dim in sample.dims if dim in used_dims]
		return (name, filtered_dims)

	filter_fns = []
	if threshold_single != None:
		filter_fns.append(filter_threshold_single)
	if threshold_both != None:
		filter_fns.append(filter_threshold_both)
	if top_sd != None:
		filter_fns.append(filter_top_sd)
	

	for filter_fn in filter_fns:
		sample.filter(filter_fn)


	return sample


def print_samples(params):
	all_cnt = 0
	for size in [150, 50, 50, 50]:
		for i in range(size):
			sample_e = load_sent(all_cnt + i, 'entailment', dict())
			sample_c = load_sent(all_cnt + i, 'contradiction', dict())
			sample_n = load_sent(all_cnt + i, 'neutral', dict())
			if i == 0:
				# check header
				print('-----')
				if sample_e.lbl != sample_e.predicted:
					print('Entailment incorrect')
				elif sample_c.lbl != sample_c.predicted:
					print('Contradiction incorrect')
				elif sample_n.lbl !=sample_n.predicted:
					print('Neutral incorrect')
				else:
					print('All correct')
				print('-----')

			print()
			print('[' + str(all_cnt +  i) + ']')
			print('[premise]', ' '.join(sample_e.p))
			print('[Entailment]', ' '.join(sample_e.h))
			print('[Contradiction]', ' '.join(sample_c.h))
			print('[Neutral]', ' '.join(sample_n.h))


		all_cnt += size


def plt_confusion_matrix(matrix, sample, title):

	# for plotting without entries
	dummy_val = matrix.max()
	label_matrix = np.copy(matrix)

	# adapt matrizes to deal with no-data fields
	matrix[matrix == LBL_NOT_DATA] = dummy_val


	fig, ax = plt.subplots()
	cax = ax.imshow(matrix, origin='upper')

	# Overwrite with colormap for empty values
	cmap = colors.ListedColormap(['white', '#0000ff00'])
	bounds=[LBL_NOT_DATA,-4000, 0]
	norm = colors.BoundaryNorm(bounds, cmap.N)
	empty = ax.imshow(label_matrix, origin='upper',cmap=cmap, norm=norm, interpolation='nearest')


	fig.colorbar(cax)
	plt.xticks(np.arange(len(sample.h)), sample.h, rotation=45)
	plt.yticks(np.arange(len(sample.p)), sample.p)
	plt.xlabel('hypothesis')
	plt.ylabel('premise')

	ax.xaxis.tick_top()
	ax.set_xlabel('hypothesis')    
	#ax.xaxis.set_label_position('top') 
	width, height = matrix.shape

	for x in range(width):
		for y in range(height):
			if label_matrix[x,y] == LBL_NOT_DATA:
				plt.annotate('-', size=6, xy=(y, x), horizontalalignment='center', verticalalignment='center')

	for x in range(width):
		for y in range(height):
			if label_matrix[x,y] != LBL_NOT_DATA:
				plt.annotate(str(label_matrix[x,y]), size=6, xy=(y, x), horizontalalignment='center', verticalalignment='center')
	
	print(title)
	print(sample.get_applied_filters())
	print('gold label:', sample.lbl)
	print('predicted:', sample.predicted)
	plt.show()


def create_conf_matrix(sample, score_fn, print_top):
	'''
	Create a confusion matrix using the score_fn for each [x,y]. The highest <print_top> value are 
	printed.
	'''
	matrix = np.zeros((len(sample.p), len(sample.h)))
	for idx_p in range(len(sample.p)):
		for idx_h in range(len(sample.h)):
			matrix[idx_p, idx_h] = score_fn(idx_p, idx_h, sample)

	# check for values to print
	sorted_values = sorted([(idx_p, idx_h, matrix[idx_p, idx_h]) for idx_p in range(len(sample.p)) for idx_h in range(len(sample.h))], key=lambda x: -x[-1])
	for idx_p, idx_h, _ in sorted_values[:print_top]:
		score_fn(idx_p, idx_h, sample, print_out=True)
	return matrix

def plt_bars(bars, title, x_labels, block=True):
	'''
	Plot a bar chart for each bar in bars.

	:param bars : list of tuples: [([labels], [values])]. Tuples will have the same bar-position per x value.
	:param title title of the bar chart
	:param x_labels appear on the x-axis
	'''


	
	labels = list(set([lbl for labels, values in bars for lbl in labels]))

	fig, ax = plt.subplots()
	bar_width = .1 

	# for each group create plots
	for i_bar, (bar_labels, values) in enumerate(bars):
		# for each label create indizes and values
		for i_lbl, lbl in enumerate(labels):
			plt_indizes = np.asarray([i for i in range(len(values)) if bar_labels[i] == lbl])
			plt_values = [values[i] for i in plt_indizes]
			plt.bar(plt_indizes + i_bar * bar_width * 2, plt_values, bar_width, color=color_palette[i_lbl], label=lbl)



	#values = [values for lbl, values in bars]
	#num_groups = len(values[0])

	#fig, ax = plt.subplots()
	num_groups = len(bars[1][0])
	index = np.arange(num_groups)
	#bar_width = .1


	#for i, lbl in enumerate(labels):
	#	plt.bar(index + i * bar_width * 2, values[i], bar_width, color=color_palette[i], label=lbl)

	plt.xlabel('Dimensions')
	plt.ylabel('Representation value')
	plt.xticks(index +  bar_width, x_labels)
	plt.title(title)

	for tick in ax.get_xticklabels():
		tick.set_rotation(90)
		tick.set_fontsize(6)

	patches = [mpatches.Patch(color=color_palette[i], label=labels[i]) for i in range(len(labels))]

	plt.legend(bbox_to_anchor=(0,1.2,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3, handles=patches)
	plt.subplots_adjust(top=0.8, bottom=0.4)
	plt.show(block=block)

def to_dim_labels(dims):
	def appendix(dim):
		known_dims = dict([(757, 'subj w/ gender'), (258, 'subj w/ gender'), (35, 'in/outdoor place+sport?'),
			(713, 'ROOT+sport'),(602, 'male'), (630, 'playable+?'), (199, 'male'), (1787, 'obj?:sport+car+animal+food?'),
			(1730, 'female'), (280, 'male'), (89, 'male'),(845, 'female'), (1840, 'obj:Location'), (311, 'female'),
			(825, 'children+play?'), (809, 'subj:m/f?'), (250, 'location'), (184, 'POS:IN/TO/(vb)'), (698, 'female?'),
			(1480, 'children+play?'), (1232, 'clothes+color'), (817, 'OBJ+(location?)?'), (1987, 'sit/eat/read/ride+=NNs?'), 
			(2007, 'action/movement VBs?'), (609, 'female'), (186, 'children+sleep?'), (1381, 'get-somewhere+rest+?'),
			(1170, 'female?'), (107, 'OBJ:/evt/sport?'), (587, 'subj:family members?'),
			(818, '1st position: The+CD'), (1878, 'subj w/ gender'), (377, 'person?'), (475, 'VB sit/ride/eat/sleep'),
			(527, 'movmentVB/NN+music'), (1341, 'sport event'), (1663, 'outsideNN+rest/walking?'), (480, 'nature/alone'),
			(262, 'man+male-assoziated'), (284, 'outside-loction'), (274, 'female+pl.persons?'), (1949, '??DT+water?'),
			(1268, 'children+/activity'), (1890, 'his/her/their?'), (778, 'run+sad/without'), (813, 'obj:water assoziated?'),
			(1893, 'CD+ads'), (977, 'male?'), (1466, 'ROOT:walk/sit/stand/+'), (1307, '???POS:DET/IN?'),
			(1251, 'ROOT:get/stay somewhere?'), (1608, 'OBJ:place?'), (513, 'POS:IN/TO'), (2037, '??POS:DET(more a)??'),
			(1903, 'POS:JJ? (city)??'), (804, '?place city?+sport'), (646, 'colors?'), (1822, '?obj?'),(1963, 'nature+activity?') ,
			(1049, 'play/dance/art?'), (140, 'places+POS:IN?'), (229, 'shopping+POS:IN'), (713, 'POS:VB, play stay(get somewhere+sport'),
			(731, 'POS:VB for sport'), (263, 'food+dogs?'), (1882, 'POS:DT+JJ(for consume?)'), (1634, 'lazy verb?'),
			(965, 'beach/water assoziated'), (980, 'POS:VB (jumpl/play/dance)'), (742, 'VB:sit/walk/stand?'),
			(495, 'ADJ to person?'), (1627, 'dog/run/police?'), (188, 'community'), (266, 'run+sport?'), (1745, 'fast running'),
			(1520, 'family:members/education/work'), (1657, '?dog/sport/food/fire?'), (45, 'city/indoor/water place'),
			(1156, 'alone/away/sad+POS:CD+JJ+RB'), (1393, 'clothing'), (1032, 'play/game/sport'), (1238, 'non-moving/emotion/commnication verbs'),
			(2044, 'water assoziated'), (756, 'location?'), (1175, 'POS:JJ'), (882, 'walk/stand:outdoor'), (2029, 'water assoziated'),
			(1223, 'indoor items/verbs'), (1370, 'water+X??'), (1174, 'JJ to (children?)'), (2020, 'adj-to-person'), 
			(957, 'children+skateboard'), (1849, 'female'), (86, 'food/animal/comfy'), (1687, 'traffic stuff??'),
			(542, 'VB: run/sleep/jump??'), (1606, 'dog/swim/run'), (1631, 'walk/sit/run/stand?'), (1812, 'water associated+food?'),
			(1329, 'dog+animals/run'), (350, 'inside+relax'), (1276, 'inside+shops/clubs'), (826, 'children+skating'),
			(1082, 'VB:work/do/perform/sing+NNs'), (1392, 'beach&biking?'), (138, 'VB:play/swim/run+sport,clown'), 
			(384, 'places:city/building/+??'), 

			(1990, 'sit+eat?'), (1449, 'people-group-genderless?'), (821, 'genderless people?'), (402, 'genderless people?'),
			(616, 'female, only few'), (1327, 'm/f subj'), (1131, 'female+children(inkl boy)'), (707, 'girl>boy>woman>man?'),
			(467, 'discuss/meet/interview??'), (1661, 'outdoor:area/weather/+?'), (2001, 'resting VBs+NNs/+?'),
			(1658, 'POS:IN+stand*'), (366, 'standing/leaning'), (941, 'standing+feet'), (1549, 'POS:DT(a/A)+CD'),
			(363, 'community/friends/events'), (828, 'SUBJ: professions'), (134, 'animals+JJ for animals'), (1607, 'people+interaction VB+NN'),
			(1720, 'crowd-ish'), (569, 'family members(female+neutral)'), (798, 'male'), (472, 'male'), (830, 'male+police+fight'),
			(842, 'male+male professions'), (1423, 'animals/people/food')
			])


		if dim in known_dims:
			return '\n' + known_dims[dim]
		else:
			return ''

	return [str(dim) + appendix(dim) for dim in dims]

def get_shared(idx_p, idx_h, sample):
	return [dim for dim in sample.dims if sample.p_act[dim] == idx_p and sample.h_act[dim] == idx_h]

# This is used when checking for not-shared dimensions. The plot will not show any information about 
# words that do not encode relevant information in terms of encoding both values >= blindthreshold
# This assumes that related words do share some dimensions with value
global blind_threshold

# This assumes that we need at least this many meaningful dimensions shared between both sentences
# Meaningful sentences are defined by reaching at least blind_threshold for both values.
global blind_threshold_min

# Of all unshared dimensions that arise from a meaningful relation defined by both thresholds above
# only those will be shown reaching a t least this threshold for one value (this value must arise from the
# word).
global min_unshared_threshold
blind_threshold = None
blind_threshold_min = None
min_unshared_threshold = None

def get_not_shared(idx_p, idx_h, sample):

	# Check if sharing enough value to be interesting
	shared_dims = get_shared(idx_p, idx_h, sample)
	interesting_shared_dims = [dim for dim in shared_dims if sample.p_rep[dim] >= blind_threshold and sample.h_rep[dim] >= blind_threshold]
	print(sample.p[idx_p], sample.h[idx_h], len(interesting_shared_dims))
	if len(interesting_shared_dims) < blind_threshold_min:
		return []

	dims_p = set([dim for dim in sample.dims if sample.p_act[dim] == idx_p and sample.p_rep[dim] >= min_unshared_threshold])
	dims_h = set([dim for dim in sample.dims if sample.h_act[dim] == idx_h and sample.h_rep[dim] >= min_unshared_threshold])
	combined = dims_p | dims_h
	# remove shared
	unshared_dims = list(combined - set(shared_dims))
	return unshared_dims


def analyse_word_alignment(params):
	idx_p = int(params['<idx_p>'])
	idx_h = int(params['<idx_h>'])
	sent_idx = int(params['<sent_idx>'])
	label = dict([('e', 'entailment'), ('n', 'neutral'), ('c', 'contradiction')])[params['<label>']]
	sample = load_sent(sent_idx, label, params)
	
	get_dims = get_shared
	if params['--not'] != None:
		get_dims = get_not_shared
		global blind_threshold
		global blind_threshold_min
		global min_unshared_threshold

		splitted = params['--not'].split(' ')
		blind_threshold = float(splitted[0])
		blind_threshold_min = int(splitted[1])
		min_unshared_threshold = float(splitted[2])

	dims = get_dims(idx_p, idx_h, sample)
	p_rep = sorted([(dim, sample.p_rep[dim]) for dim in dims], key=lambda x: -x[-1])
	h_rep = [sample.h_rep[dim] for dim, _ in p_rep]
	dims = [dim for dim, _ in p_rep]
	
	labels_p = ['[p' + str(sample.p_act[dim]) + ']' + sample.p[sample.p_act[dim]] for dim,_ in p_rep]
	labels_h = ['[h' + str(sample.h_act[dim]) + ']' + sample.h[sample.h_act[dim]] for dim,_ in p_rep]
	p_rep = [v for dim, v in p_rep]
	bars = [(labels_p, p_rep), (labels_h, h_rep)]
	x_labels = to_dim_labels(dims)
	title = '[premise=' + str(idx_p) + ']' + sample.p[idx_p] + '  --  [hypothesis=' + str(idx_h) + ']' + sample.h[idx_h]
	plt_bars(bars, title, x_labels, block=True)



def analyse_sent_alignment(params):
	
	sent_idx = int(params['<sent_idx>'])
	label = dict([('e', 'entailment'), ('n', 'neutral'), ('c', 'contradiction')])[params['<label>']]
	sample = load_sent(sent_idx, label, params)
	conf_type = params['<conf_type>']
	print_top = int(params['--pt'] or 1)

	get_dims = get_shared
	if params['--not'] != None:
		get_dims = get_not_shared
		global blind_threshold
		global blind_threshold_min
		global min_unshared_threshold

		splitted = params['--not'].split(' ')
		blind_threshold = float(splitted[0])
		blind_threshold_min = int(splitted[1])
		min_unshared_threshold = float(splitted[2])

	def score_num_act(idx_p, idx_h, sample, print_out=False):
		'''Score each index by the amount of activation they share.'''
		act_dims = get_dims(idx_p, idx_h, sample)
		if print_out:
			print('Shared dimensions for', sample.p[idx_p] + '(premise)', sample.h[idx_h] + '(hypothesis):', len(act_dims))
			print(act_dims)
		return len(act_dims)

	def score_mean_diff(idx_p, idx_h, sample, print_out=False):
		act_dims = get_dims(idx_p, idx_h, sample)
		values = [np.absolute(sample.p_rep[dim] - sample.h_rep[dim]) for dim in act_dims]
		if len(values) == 0:
			return LBL_NOT_DATA

		mean_diff = sum(values) / len(values)

		if print_out:
			print('Mean difference for', sample.p[idx_p] + '(premise)', sample.h[idx_h] + '(hypothesis):', mean_diff)
			print(sorted([(act_dims[i], values[i]) for i in range(len(act_dims))], key=lambda x: -x[-1]))
		return round(mean_diff, 3)

	def score_max_diff(idx_p, idx_h, sample, print_out=False):
		act_dims = get_dims(idx_p, idx_h, sample)
		values = [np.absolute(sample.p_rep[dim] - sample.h_rep[dim]) for dim in act_dims]
		if len(values) == 0:
			return LBL_NOT_DATA

		max_diff = max(values)
		if print_out:
			print('Max difference for', sample.p[idx_p] + '(premise)', sample.h[idx_h] + '(hypothesis):', max_diff)
			print(sorted([(act_dims[i], values[i]) for i in range(len(act_dims))], key=lambda x: -x[-1]))
		return round(max_diff, 3)

	def score_mean_common(idx_p, idx_h, sample, print_out=False):
		act_dims = get_dims(idx_p, idx_h, sample)
		values = [min([sample.p_rep[dim], sample.h_rep[dim]]) for dim in act_dims]
		if len(values) == 0:
			return LBL_NOT_DATA
		mean_common = sum(values) / len(values)
		if print_out:
			print('Mean common value', sample.p[idx_p] + '(premise)', sample.h[idx_h] + '(hypothesis):', mean_common)
			print(sorted([(act_dims[i], values[i]) for i in range(len(act_dims))], key=lambda x: -x[-1]))
		return round(mean_common, 3)

	def score_max_common(idx_p, idx_h, sample, print_out=False):
		act_dims = get_dims(idx_p, idx_h, sample)
		values = [min([sample.p_rep[dim], sample.h_rep[dim]]) for dim in act_dims]
		if len(values) == 0:
			return LBL_NOT_DATA
		max_common = max(values)
		if print_out:
			print('Max common value', sample.p[idx_p] + '(premise)', sample.h[idx_h] + '(hypothesis):', max_common)
			print(sorted([(act_dims[i], values[i]) for i in range(len(act_dims))], key=lambda x: -x[-1]))
		return round(max_common, 3)

	def score_mean_product(idx_p, idx_h, sample, print_out=False):
		act_dims = get_dims(idx_p, idx_h, sample)
		values = [sample.p_rep[dim] * sample.h_rep[dim] for dim in act_dims]
		if len(values) == 0:
			return LBL_NOT_DATA

		mean_product = sum(values) / len(values)
		if print_out:
			print('mean product', sample.p[idx_p] + '(premise)', sample.h[idx_h] + '(hypothesis):', mean_product)
			print(sorted([(act_dims[i], values[i]) for i in range(len(act_dims))], key=lambda x: -x[-1]))
		return round(mean_product, 3)

	def score_max_product(idx_p, idx_h, sample, print_out=False):
		act_dims = get_dims(idx_p, idx_h, sample)
		values = [sample.p_rep[dim] * sample.h_rep[dim] for dim in act_dims]
		if len(values) == 0:
			return LBL_NOT_DATA

		max_product = max(values)
		if print_out:
			print('max product', sample.p[idx_p] + '(premise)', sample.h[idx_h] + '(hypothesis):', max_product)
			print(sorted([(act_dims[i], values[i]) for i in range(len(act_dims))], key=lambda x: -x[-1]))
		return round(max_product, 3)

	def score_min_product(idx_p, idx_h, sample, print_out=False):
		act_dims = get_dims(idx_p, idx_h, sample)
		values = [sample.p_rep[dim] * sample.h_rep[dim] for dim in act_dims]
		if len(values) == 0:
			return LBL_NOT_DATA

		min_product = sum(values) / len(values)
		if print_out:
			print('min product', sample.p[idx_p] + '(premise)', sample.h[idx_h] + '(hypothesis):', min_product)
			print(sorted([(act_dims[i], values[i]) for i in range(len(act_dims))], key=lambda x: -x[-1]))
		return round(min_product, 3)

	fn_dict = dict()
	fn_dict['nshared'] = ('Amount of same dimension per word', score_num_act)
	fn_dict['meandiff'] = ('Mean difference of shared dimensions', score_mean_diff)
	fn_dict['maxdiff'] = ('Maximum difference of shared dimensions', score_max_diff)
	fn_dict['meanc'] = ('Mean of the value reached by both dimensions', score_mean_common)
	fn_dict['maxc'] = ('Max of the value reached by both dimensions', score_max_common)
	fn_dict['meanprod'] = ('Mean value reached by both dimensions multiplicated', score_mean_product)
	fn_dict['maxprod'] = ('Max value reached by both dimensions multiplicated', score_max_product)
	fn_dict['minprod'] = ('Min value reached by both dimensions multiplicated', score_min_product)



	title, fn = fn_dict.get(conf_type, ('In case invalid function', None))
	if fn == None:
		print('Choose one of the following:')
		print(fn_dict.keys())
		return

	matrix = create_conf_matrix(sample, fn, print_top)
	plt_confusion_matrix(matrix, sample, title)

mapper = dict()
mapper['generate_data'] = generate_data
mapper['sd'] = print_sd_rank
mapper['cm'] = analyse_sent_alignment
mapper['print'] = print_samples
mapper['plt'] = analyse_word_alignment

def main():
	args = docopt("""Analyse the alignment between premise and hypothesis.
		conf_type can be:
		nshared - number of shared dimension
		meandiff - mean difference of all values of shared dimensions
		maxdiff - max difference of all values of shared dimensions
		meanc -  mean of what is the value that both sentences reach (lower value)
		maxc - 	same but as max
		meanprod - mean product of the common dimensions
		maxprod - maximum product of the common dimensions
		minprod - minimum product of the common dimensions


	Usage:
		analyse-alignment.py generate_data <folder>
		analyse-alignment.py print <folder>
		analyse-alignment.py sd <folder>		
		analyse-alignment.py cm <sent_idx> <label> <conf_type> [--pt=<print_top>] [--t=<threshold>] [--tb=<threshold_both>] [--tsd=<top_sd>] [--not=<blind_threshold>]
		analyse-alignment.py plt <sent_idx> <label> <conf_type> <idx_p> <idx_h> [--t=<threshold>] [--tb=<threshold_both>] [--tsd=<top_sd>] [--not=<blind_threshold>]
	""")

	fn = [k for k in args if args[k] == True][0]

	mapper[fn](args)


if __name__ == '__main__':
	main()
#