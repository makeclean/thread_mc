#include <iostream>
#include "moab/Core.hpp"
#include "DagMC.hpp"
#include "math.h"

#include "test.hpp"

#ifdef _OPENMP
  #include "omp.h"
#else
  #define omp_get_thread_num() 0
  #define omp_get_num_threads() 1
#endif

moab::DagMC *dag = moab::DagMC::instance();

moab::EntityHandle find_location(double x, double y, double z);

std::uniform_real_distribution<double> uni(0,1.);

generator_store::generator_store(int num_threads,int stride) {
  generators.reserve(num_threads);
  for ( unsigned int i = 0 ; i < num_threads ; i++ ) {
    std::default_random_engine gen;
    generators[i] = gen;
  }
  this->stride = stride;
}

generator_store::~generator_store() {
}

void generator_store::reseed(int thread_num, int nps) {
  int seed = nps*stride;
  generators[thread_num].seed(seed); 
}

double generator_store::sample(int thread_num) {
  return uni(generators[thread_num]);
}

int main(int argc, char* argv[]){

  moab::ErrorCode rval;
  rval = dag->load_file("test_slabs.h5m");
  if(rval != moab::MB_SUCCESS) return -1;
  rval = dag->init_OBBTree();
  if(rval != moab::MB_SUCCESS) return -1;

  double x = 2.5;
  double y = 0.0;
  double z = 1.0;
  
  double t_l[15];

  generator_store *rn = new generator_store(omp_get_num_threads(),12345);

  unsigned int nps = 1000000;
  // loop over the history like loop
  #pragma omp parallel for  \
   private(rval) shared(t_l) 
  // #pragma omp parallel for
  for ( unsigned int i = 0 ; i < nps ; i++ ) {

    // each thread gets rn generator
    rn->reseed(omp_get_thread_num(),i);
    // make each seed unique
    //srand(i*123456);

    bool alive = true;
    moab::EntityHandle volume;
    double pos[3] = {x,y,z};
    //    double dir[3] = {1,0,0};
    double dir[3];
    dir[0] = rn->sample(omp_get_thread_num()); //rand();
    dir[1] = rn->sample(omp_get_thread_num()); //rand();
    dir[2] = rn->sample(omp_get_thread_num()); // rand();

    double magnitude = sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
    dir[0] = dir[0]/magnitude;
    dir[1] = dir[1]/magnitude;
    dir[2] = dir[2]/magnitude;

    // determine the start volume
    volume = find_location(x,y,z);
    while (alive) {
      moab::EntityHandle surf;
      double dist;
      if(dag->is_implicit_complement(volume)){
	alive = false;
	break;
      } else {
	rval = dag->ray_fire(volume,pos,dir,surf,dist);
	if(surf == 0 ) {
	  alive = false;
	  break;
	}
	rval = dag->next_vol(surf,volume,volume);
	pos[0] += dist*dir[0];
	pos[1] += dist*dir[1];
	pos[2] += dist*dir[2];
	
	int cel = dag->index_by_handle(volume);
        #pragma omp atomic
	t_l[cel] += dist;
      }
    }
  }

  for ( unsigned int i = 0 ; i < 10 ; i++ ) {
    std::cout << i+1 << " " << t_l[i+1]/(double)nps << std::endl;
  }
  return 0;
}

// determine the particle location
moab::EntityHandle find_location(double x, double y, double z) {
  moab::ErrorCode rval;
  moab::EntityHandle volume;
  int inside;
  double pos[3];
  pos[0]=x;pos[1]=y;pos[2]=z;
  for ( unsigned int i = 0 ; i < dag->num_entities(3) ; i++ ) {
    volume = dag->entity_by_index(3,i+1);
    rval = dag->point_in_volume(volume,pos,inside); 
    if(inside == 1) return volume;
  }
  return 0;
}

