#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

#include <boost/tuple/tuple.hpp>

using boost::get;
using boost::make_tuple;
using boost::tuple;
using std::vector;

namespace pop
{

// Ralph Bucher and D. Misra, “A Synthesizable VHDL Model of the Exact Solution
// for Three-dimensional Hyperbolic Positioning System,” VLSI Design, vol. 15,
// no. 2, pp. 507-520, 2002. doi:10.1080/1065514021000012129

tuple<double, double, double> calculate_xyz_position(
	const vector<tuple<double, double, double, double> >& sets)
{
	double ti=get<3>(sets[0])*1000000000.0; double tk=get<3>(sets[2])*1000000000.0; double tj=get<3>(sets[1])*1000000000.0; double tl=get<3>(sets[3])*1000000000.0;
	double xi=get<0>(sets[0]); double xk=get<0>(sets[2]); double xj=get<0>(sets[1]); double xl=get<0>(sets[3]);
	double yi=get<1>(sets[0]); double yk=get<1>(sets[2]); double yj=get<1>(sets[1]); double yl=get<1>(sets[3]);
	double zi=get<2>(sets[0]); double zk=get<2>(sets[2]); double zj=get<2>(sets[1]); double zl=get<2>(sets[3]);

	printf("ti = %.6f\n", ti);      printf("tj = %.6f\n", tj);      printf("tk = %.6f\n", tk);
	printf("tl = %.6f\n", tl);      printf("xi = %.6f\n", xi);      printf("xj = %.6f\n", xj);
	printf("xk = %.6f\n", xk);      printf("xl = %.6f\n", xl);      printf("yi = %.6f\n", yi);
	printf("yj = %.6f\n", yj);      printf("yk = %.6f\n", yk);      printf("yl = %.6f\n", yl);
	printf("zi = %.6f\n", zi);      printf("zj = %.6f\n", zj);      printf("zk = %.6f\n", zk);
	printf("zl = %.6f\n", zl);

	double xji=xj-xi; double xki=xk-xi; double xjk=xj-xk; double xlk=xl-xk;
	double xik=xi-xk; double yji=yj-yi; double yki=yk-yi; double yjk=yj-yk;
	double ylk=yl-yk; double yik=yi-yk; double zji=zj-zi; double zki=zk-zi;
	double zik=zi-zk; double zjk=zj-zk; double zlk=zl-zk;

	double rij=abs((100000*(ti-tj))/333564); double rik=abs((100000*(ti-tk))/333564);
	double rkj=abs((100000*(tk-tj))/333564); double rkl=abs((100000*(tk-tl))/333564);

	double s9 =rik*xji-rij*xki; double s10=rij*yki-rik*yji; double s11=rik*zji-rij*zki;
	double s12=(rik*(rij*rij + xi*xi - xj*xj + yi*yi - yj*yj + zi*zi - zj*zj)
	           -rij*(rik*rik + xi*xi - xk*xk + yi*yi - yk*yk + zi*zi - zk*zk))/2;

	double s13=rkl*xjk-rkj*xlk; double s14=rkj*ylk-rkl*yjk; double s15=rkl*zjk-rkj*zlk;
	double s16=(rkl*(rkj*rkj + xk*xk - xj*xj + yk*yk - yj*yj + zk*zk - zj*zj)
	           -rkj*(rkl*rkl + xk*xk - xl*xl + yk*yk - yl*yl + zk*zk - zl*zl))/2;

	double a= s9/s10; double b=s11/s10; double c=s12/s10; double d=s13/s14;
	double e=s15/s14; double f=s16/s14; double g=(e-b)/(a-d); double h=(f-c)/(a-d);
	double i=(a*g)+b; double j=(a*h)+c;
	double k=rik*rik+xi*xi-xk*xk+yi*yi-yk*yk+zi*zi-zk*zk+2*h*xki+2*j*yki;
	double l=2*(g*xki+i*yki+2*zki);
	double m=4*rik*rik*(g*g+i*i+1)-l*l;
	double n=8*rik*rik*(g*(xi-h)+i*(yi-j)+zi)+2*l*k;
	double o=4*rik*rik*((xi-h)*(xi-h)+(yi-j)*(yi-j)+zi*zi)-k*k;
	double s28=n/(2*m);     double s29=(o/m);       double s30=(s28*s28)-s29;
	double root=sqrt(s30);        printf("s30 = %.6f\n", s30);
	double z1=s28+root;           printf("z1 = %.6f\n", z1);
	double z2=s28-root;           printf("z2 = %.6f\n", z2);
	double x1=g*z1+h;             printf("x1 = %.6f\n", x1);
	double x2=g*z2+h;             printf("x2 = %.6f\n", x2);
	double y1=a*x1+b*z1+c;        printf("y1 = %.6f\n", y1);
	double y2=a*x2+b*z2+c;        printf("y2 = %.6f\n", y2);

	return make_tuple(x2, y2, z2);
}

}
