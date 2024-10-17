#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>

double wallclock()
{
  struct timeval tv;
  struct timezone tz;
  double t;

  gettimeofday(&tv, &tz);

  t = (double)tv.tv_sec;
  t += ((double)tv.tv_usec)/1000000.0;

  return t;
}
