.PHONY: all check clean

all:
	$(MAKE) -C v9

check:
	$(MAKE) -C v9 check

clean:
	$(MAKE) -C v9 clean
