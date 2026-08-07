# wildfire-nowcast — developer tasks.
#
# synth/movie target the intended CLI entry points; the modules they invoke
# are implemented later by the leads (no science code exists yet).

OUT    ?= outputs/synthetic_fire.zarr
TENSOR ?= outputs/synthetic_fire.zarr
MOVIE  ?= outputs/fire.mp4

.PHONY: test lint synth movie

## test: run the test suite
test:
	pytest

## lint: lint (and check formatting of) src and tests
lint:
	ruff check src tests

## synth: generate one synthetic fire -> $(OUT)
synth:
	python -m wildfire_nowcast.common.synthetic --out $(OUT)

## movie: render a fire movie from a tensor path -> $(MOVIE)
movie:
	python -m wildfire_nowcast.sim.movie --tensor $(TENSOR) --out $(MOVIE)
