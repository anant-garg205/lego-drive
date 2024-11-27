# LeGo-Drive: Language-enhanced Goal-oriented <br> Closed-Loop End-to-End Autonomous Driving

[Pranjal Paul](https://scholar.google.com/citations?user=4pZlCV4AAAAJ&hl=en) <sup>\*1</sup>,
[Anant Garg](https://anant-garg205.github.io/) <sup>\*1</sup>,
[Tushar Chaudhary]() <sup>1</sup>,
[Arun Kumar Singh](https://tuit.ut.ee/en/content/arun-kumar-singh) <sup>2</sup>,
[Madhava Krishna](https://www.iiit.ac.in/people/faculty/mkrishna/) <sup>1</sup>,

<sup>1</sup>International Institute of Information Technology, Hyderabad, <sup>2</sup>University of Tartu

<sup>\*</sup>denotes equal contribution

![Teaser](/assets/teaser-wide.png)

<table align="center" border="0">
    <tr>
        <td align="center">
            <a href="https://reachpranjal.github.io/lego-drive" style="text-decoration: none;">
                <img src="https://img.shields.io/badge/Project_Page-4CAF50?style=for-the-badge&logoColor=white&logo=github" alt="GitHub Logo">
            </a>
        </td>
        <td align="center">
            <a href="https://arxiv.org/abs/2403.20116" style="text-decoration: none;">
                <img src="https://img.shields.io/badge/ArXiv-000000?style=for-the-badge&logoColor=white&logo=arxiv" alt="ArXiv Logo">
            </a>
        </td>
        <td align="center">
            <a href="https://www.youtube.com/watch?v=eOYAq2cz1Pk" style="text-decoration: none;">
                <img src="https://img.shields.io/badge/Demo_Video-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube Logo">
            </a>
        </td>
    </tr>
</table>

<hr>

## Setup
Download and setup CARLA 0.9.15
```
mkdir carla
cd carla
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.15.tar.gz
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.15.tar.gz
tar -xf CARLA_0.9.15.tar.gz
tar -xf AdditionalMaps_0.9.15.tar.gz
rm CARLA_0.9.15.tar.gz
rm AdditionalMaps_0.9.15.tar.gz
cd ..
```

## Run demo
This will load a pre-recorded scenario in CARLA and will prompt the user to type in a language command.
```
python3 navbev/sim/run_demo.py
```


### Cite Us
```
@article{paul2024lego,
  title={LeGo-Drive: Language-enhanced Goal-oriented Closed-Loop End-to-End Autonomous Driving},
  author={Paul, Pranjal and Garg, Anant and Choudhary, Tushar and Singh, Arun Kumar and Krishna, K Madhava},
  journal={arXiv preprint arXiv:2403.20116},
  year={2024}
}
```

## Contact

Pranjal Paul: [pranjal.paul@research.iiit.ac.in](pranjal.paul@research.iiit.ac.in) <br>
Anant Garg: [garg.anant205@gmail.com](garg.anant205@gmail.com)

