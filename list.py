import re

regex = r"^\.MODEL (.+) D$"
lista = []
string = open("./netlist/diodo_teste.lib", "r")
for word in string.readlines():
	match=re.match(regex, word)
	if match:
		teste = str(match[1])
		if teste != 'DIODE1': 
			if teste != 'DIODE2':
				lista.append(match[1])
			
print (len(lista))

			
	
	
