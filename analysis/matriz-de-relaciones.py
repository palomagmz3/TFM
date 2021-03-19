# This is a sample Python script.
from program_hashtags import hashtags_L6N20151024, \
    hashtags_L6N20151114, \
    hashtags_L6N20151031, \
    hashtags_L6N20151107, \
    hashtags_L6N20151121, \
    hashtags_L6N20151128, \
    hashtags_L6N20151205, \
    hashtags_L6N20151212, \
    hashtags_L6N20151219, \
    hashtags_L6N20151226, \
    hashtags_L6N20160102, \
    hashtags_L6N20160109, \
    hashtags_L6N20160116, \
    hashtags_L6N20160123

#Cambiar NodeNames para ver la matriz de el programa desesado
NodeNames = hashtags_L6N20160116

#Cambiar el extremo final de la ruta al fichero txt donde están los grupos de hashtags y cuántas veces aparecen juntos
with open('/Users/palomagomez/Documents/Documents/Teleco/TFM/matriz-de-relaciones/L6N20160116.txt') as f:
    lines = f.readlines()

num_lines = len(lines)
num_tags = len(NodeNames)
matrix = []
line_count = []
for i in range(num_lines):
    tokens = lines[i].strip().split(': ')
    a = [int(s) for s in tokens if s.isdigit()]
    if len(a) == 1:
        line_count.append(a[0])
    else:
        print(len(a))
        print(tokens)
        print('PROBLEMS!!!')

matrix = [[0 for i in range(num_tags)] for j in range(num_tags)]

for i in range(num_lines):
    for j in range(num_tags):
        if NodeNames[j] in lines[i]:
            for k in range(num_tags):
                if j != k:
                    if NodeNames[k] in lines[i]:
                        matrix[j][k] = matrix[j][k] + line_count[i]

print('\t', end='')
for i in range(num_tags):
    print('%s\t' % NodeNames[i], end='')
print()
for i in range(num_tags):
    print('%s\t' % NodeNames[i], end='')
    for j in range(num_tags):
        print('%d\t' % matrix[i][j], end='')
    print()
