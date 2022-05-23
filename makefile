exp:
	python experiment1_9higher_part1.py
	python experiment2_9higher_part2.py
	python experiment3_9higher_part3.py
	python experiment4_9lower.py

exp_set:
	python experiment0_set_featboot.py

analysis:
	python -W ignore analysis5_cross_val_in_opt.py
	cd article && pdflatex main.tex main.pdf && xdg-open main.pdf
exp_cv:
	python experiment5_cross_val_in_opt_9h.py
	python experiment5_cross_val_in_opt_9l.py
