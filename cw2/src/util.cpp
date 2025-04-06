// Copyright (C) 2025, Rupert Nash, The University of Edinburgh.
//
// All rights reserved.
//
// This file is provided to you to complete an assessment and for
// subsequent private study. It may not be shared and, in particular,
// may not be posted on the internet. Sharing this or any modified
// version may constitute academic misconduct under the University's
// regulations.

#include "util.h"

#include <algorithm>
#include <random>
#include <png.h>

#include "decomp.h"

// Print the map to the file. If bounds is true, include the boundary
// halo.
void txt_print(FILE* f, int M, int N, int const* map, int bounds) {
  int lo_i = bounds ? 0 : 1;
  int hi_i = bounds ? M+2 : M+1;
  int lo_j = bounds ? 0 : 1;
  int hi_j = bounds ? N+2 : N+1;
  int stride = N + 2;

  for (int j = lo_j; j < hi_j; ++j) {
    std::fprintf(f, "%3d", map[lo_i * stride + j]);
    for (int i = lo_i + 1; i < hi_i; ++i) {
      std::fprintf(f, " %3d", map[i * stride + j]);
    }
    std::fprintf(f, "\n");
  }
}

// Initialise map with target porosity. Zero indicates rock, a
// positive value indicates a hole. For the algorithm to work, all the
// holes must be initialised with a unique integer. Needs to know the
// total problem size, the subdomain size and the index of this
// subdomain.
//
// Returns number of holes on this process.
int fill_map(unsigned seed, float porosity,
	     ParallelDecomp* pp,
	     int* map) {
  auto& par = *pp;
  int lhole = 0;
  std::mt19937 gen{seed};
  std::uniform_real_distribution<float> uni;

  // Zero based index limits for this process
  int const gi_lo = par.pi * par.sub_nx;
  int const gi_hi = std::min(par.nx, (par.pi + 1) * par.sub_nx);
  // local count in i
  int const lci = gi_hi - gi_lo;
  // Zero based index limits for this process
  int const gj_lo = par.pj * par.sub_ny;
  int const gj_hi = std::min(par.ny, (par.pj + 1) * par.sub_ny);
  // local count in j
  int const lcj = gj_hi - gj_lo;

  int const lstride = par.sub_ny + 2;

  // Zero edges, will have to communicate to initialise if they are
  // real sites.
  for (int lj = 0; lj < lcj + 2; ++lj) {
    // i = 0
    map[0*lstride + lj] = 0;
    // i = M + 1
    map[(lci + 1)*lstride + lj] = 0;
  }
  for (int li = 1; li < lci + 2; ++li) {
    // j = 0
    map[li*lstride + 0] = 0;
    // j = N + 1
    map[li*lstride + lcj + 1] = 0;
  }

  // Fill middle

  // First, advance PRNG state to this global row
  gen.discard(par.pi * par.sub_nx * par.ny);
  // Then, advance to the start of the subdomain within it
  gen.discard(par.pj * par.sub_ny);
  for (int li = 0; li < lci; ++li) {
    int const gi = par.pi * par.sub_nx + li;

    for (int lj = 0; lj < lcj; ++lj) {
      int const gj = par.pj * par.sub_ny + lj;
      auto r = uni(gen);

      int val = 0;
      if (r < porosity) {
	++lhole;
	// The plus one is to allow site (0,0) to be solid
	val = gi * par.ny + gj + 1;
      }
      map[(li + 1) * lstride + lj + 1] = val;
    }
    // Advance from the end of this row in subdomain to the start of
    // the next
    gen.discard(par.ny - lcj);
  }
  return lhole;
}


// Convert HSV(hue, 1, 1) to RGB color space
template <typename T>
static void hue2rgb(float hue, T rgb[3]) {
  constexpr auto MAX = std::numeric_limits<T>::max();
  float const huePrime = std::fmod(6.0f * hue, 6.0f);
  T const fX = (1.f - std::fabs(std::fmod(huePrime, 2.f) - 1.f)) * MAX;

  if(0 <= huePrime && huePrime < 1) {
    rgb[0] = MAX;
    rgb[1] = fX;
    rgb[2] = 0;
  } else if(1 <= huePrime && huePrime < 2) {
    rgb[0] = fX;
    rgb[1] = MAX;
    rgb[2] = 0;
  } else if(2 <= huePrime && huePrime < 3) {
    rgb[0] = 0;
    rgb[1] = MAX;
    rgb[2] = fX;
  } else if(3 <= huePrime && huePrime < 4) {
    rgb[0] = 0;
    rgb[1] = fX;
    rgb[2] = MAX;
  } else if(4 <= huePrime && huePrime < 5) {
    rgb[0] = fX;
    rgb[1] = 0;
    rgb[2] = MAX;
  } else if(5 <= huePrime && huePrime < 6) {
    rgb[0] = MAX;
    rgb[1] = 0;
    rgb[2] = fX;
  } else {
    rgb[0] = 0;
    rgb[1] = 0;
    rgb[2] = 0;
  }
}

int write_state_png(char const* file_name, ParallelDecomp* pp, int const* state) {
  auto& par = *pp;

  bool const root = par.rank == 0;
  std::FILE *fp = nullptr;
  png_structp png_ptr = nullptr;
  png_infop info_ptr = nullptr;

  if (root) {
    fp = std::fopen(file_name, "wb");
    if (!fp) {
      std::fprintf(stderr, "Could not open file '%s'\n", file_name);
      MPI_Abort(par.comm, 1);
    }

    // PNG basic init
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr,
				      nullptr, nullptr);
    if (!png_ptr) {
      std::fprintf(stderr, "png_create_write_struct error");
      std::fclose(fp);
      MPI_Abort(par.comm, 1);
    }
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
      std::fprintf(stderr, "png_create_info_struct error");
      png_destroy_write_struct(&png_ptr,
			       nullptr);
      std::fclose(fp);
      MPI_Abort(par.comm, 1);
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
      std::fprintf(stderr, "PNG error in png_init_io\n");
      png_destroy_write_struct(&png_ptr, &info_ptr);
      std::fclose(fp);
      MPI_Abort(par.comm, 1);
    }
    png_init_io(png_ptr, fp);

    // Headers
    if (setjmp(png_jmpbuf(png_ptr))) {
      std::fprintf(stderr, "PNG error in during header write\n");
      png_destroy_write_struct(&png_ptr, &info_ptr);
      std::fclose(fp);
      MPI_Abort(par.comm, 1);
    }
    // NOTE: using y/j coord for the image width!
    // ALSO: using 8 bit colour depth to avoid endian issues
    png_set_IHDR(png_ptr, info_ptr, par.ny, par.nx,
		 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
		 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(png_ptr, info_ptr);

    // write data
    if (setjmp(png_jmpbuf(png_ptr))) {
      std::fprintf(stderr, "PNG error in during data write\n");
      png_destroy_write_struct(&png_ptr, &info_ptr);
      std::fclose(fp);
      MPI_Abort(par.comm, 1);
    }
  }

  // Write row by row
  // 8 bits per channel
  // Write rock/solid (== 0) as black.
  // Map fluid (1 <= x <= maxid) onto hue (0, 1.0), then convert HSV
  // (h, 1, 1) into RGB.
  int maxid = par.nx * par.ny;

  // Only actually used on rank 0
  std::vector<std::uint8_t> grow;
  std::vector<MPI_Request> recvs;
  if (root) {
    grow.resize(3 * par.sub_ny * par.py);
    recvs.resize(par.py);
  }
  // This is used on all
  std::vector<std::uint8_t> lrow(3 * par.sub_ny);

  // Loop over rows
  // First, by the block/subdomain
  for (int block_i = 0; block_i < par.px; ++block_i) {
    for (int li = 0; li < par.sub_nx; ++li) {
      int gi = block_i * par.sub_nx + li;
      if (gi >= par.nx)
	// This row is beyond the image, we're done!
	break;

      // First set recvs on rank 0
      if (root) {
	auto buf = grow.data();
	for (int block_j = 0; block_j < par.py; ++block_j) {
	  MPI_Irecv(buf + 3 * par.sub_ny * block_j,
		    3 * par.sub_ny, MPI_UINT8_T,
		    par.global_rank(block_i, block_j), 0,
		    par.comm,
		    &recvs[block_j]);
	}
      }

      // Convert and send data if we hold it
      if (par.pi == block_i) {
	for (int lj = 0; lj < par.sub_ny; ++lj) {
	  int const p = state[(li + 1) * (par.sub_ny + 2) + lj + 1];
	  int r = 3 * lj;
	  if (p == 0) {
	    // solid is black
	    lrow[r + 0] = 0;
	    lrow[r + 1] = 0;
	    lrow[r + 2] = 0;
	  } else {
	    hue2rgb(float(p - 1) / float(maxid), &lrow[r]);
	  }
	}

	MPI_Ssend(lrow.data(),
		  3 * par.sub_ny, MPI_UINT8_T,
		  0, 0,
		  par.comm);
      }

      // Root waits for comms then writes
      if (root) {
	MPI_Waitall(par.py, recvs.data(), MPI_STATUSES_IGNORE);
	png_write_row(png_ptr, reinterpret_cast<unsigned char const*>(grow.data()));
      }

    }
  }

  // Finish writing, close file, etc.
  if (root) {
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    std::fclose(fp);
  }

  return 0;
}
