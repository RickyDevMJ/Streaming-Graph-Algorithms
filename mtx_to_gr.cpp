#include <bits/stdc++.h>
using namespace std;

int main()
{
	string temp;
	while(cin.peek() == '%')
	{
		getline(cin, temp);
	}
	int n, m;
	cin >> n >> n >> m;
	cout << "p sp " << n << " " << m << endl;
	
	for(int i=0; i<m; i++)
	{
		while(cin.peek() == '%')
		{
			getline(cin, temp);
		}
		int u, v;
		cin >> u >> v;
		cout << "a " << u << " " << v << " 1\n";
	}
	
	
	return 0;
}
